#!/usr/bin/env python3
"""
Production-Ready Stock Data Pipeline Scheduler
Automatically runs stock data gathering and feature engineering every 24 hours
"""

import schedule
import time
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Get the directory where scheduler.py is located
    BASE_DIR = Path(__file__).parent.absolute()
    
    # File paths (all relative to BASE_DIR)
    STOCK_DATA_SCRIPT = str(BASE_DIR / "stock_data_gathering.py")
    STOCK_VECTOR_SCRIPT = str(BASE_DIR / "stock_vector.py")
    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE = LOG_DIR / "scheduler.log"
    STATUS_FILE = LOG_DIR / "pipeline_status.json"
    
    # Scheduling
    SCHEDULE_TIME = "02:00"  # Run at 2 AM daily (low market activity)
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 300  # 5 minutes between retries
    
    # Timeout configuration
    GATHERING_TIMEOUT = 7200  # 2 hours max for data gathering
    VECTOR_TIMEOUT = 1800     # 30 minutes max for feature engineering


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure production-grade logging"""
    Config.LOG_DIR.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (all logs)
    file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_status(status: dict):
    """Save pipeline execution status to JSON file"""
    try:
        with open(Config.STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save status file: {e}")


def load_last_status() -> Optional[dict]:
    """Load last pipeline execution status"""
    try:
        if Config.STATUS_FILE.exists():
            with open(Config.STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load last status: {e}")
    return None


def run_script(script_path: str, timeout: int, script_name: str) -> Tuple[bool, str]:
    """
    Run a Python script with timeout and capture output
    
    Args:
        script_path: Path to the Python script
        timeout: Maximum execution time in seconds
        script_name: Human-readable script name for logging
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    logger.info(f"{'='*70}")
    logger.info(f"Starting: {script_name}")
    logger.info(f"Script: {script_path}")
    logger.info(f"Timeout: {timeout}s ({timeout//60} minutes)")
    logger.info(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Check if script exists
        if not Path(script_path).exists():
            error_msg = f"Script not found: {script_path}"
            logger.error(error_msg)
            return False, error_msg
        
        # Prepare environment with UTF-8 encoding
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the script with UTF-8 encoding
        result = subprocess.run(
            [sys.executable, '-u', script_path],  # -u for unbuffered output
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # Don't raise exception on non-zero exit
            env=env,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        
        elapsed_time = time.time() - start_time
        
        # Log output
        if result.stdout:
            logger.debug(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR:\n{result.stderr}")
        
        # Check exit code
        if result.returncode == 0:
            logger.info(f"✓ {script_name} completed successfully")
            logger.info(f"  Execution time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
            return True, result.stdout
        else:
            error_msg = f"Script exited with code {result.returncode}"
            logger.error(f"✗ {script_name} failed: {error_msg}")
            logger.error(f"  Execution time: {elapsed_time:.1f}s")
            return False, f"{error_msg}\n{result.stderr}"
    
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        error_msg = f"Script timed out after {timeout}s"
        logger.error(f"✗ {script_name} failed: {error_msg}")
        logger.error(f"  Execution time: {elapsed_time:.1f}s")
        return False, error_msg
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"✗ {script_name} failed: {error_msg}")
        logger.error(f"  Execution time: {elapsed_time:.1f}s")
        logger.debug(traceback.format_exc())
        return False, error_msg


def run_with_retry(script_path: str, timeout: int, script_name: str, 
                   max_retries: int = Config.MAX_RETRIES) -> Tuple[bool, str]:
    """
    Run script with automatic retry on failure
    
    Args:
        script_path: Path to the Python script
        timeout: Maximum execution time in seconds
        script_name: Human-readable script name
        max_retries: Maximum number of retry attempts
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries} for {script_name}")
        
        success, output = run_script(script_path, timeout, script_name)
        
        if success:
            return True, output
        
        if attempt < max_retries:
            logger.warning(f"Retry scheduled in {Config.RETRY_DELAY_SECONDS}s...")
            time.sleep(Config.RETRY_DELAY_SECONDS)
        else:
            logger.error(f"All {max_retries} attempts failed for {script_name}")
    
    return False, output


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    """
    Execute the complete stock data pipeline:
    1. Fetch stock data (stock_data_gathering.py)
    2. Generate feature vectors (stock_vector.py)
    """
    pipeline_start = datetime.now()
    status = {
        "start_time": pipeline_start.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "data_gathering": {"success": False, "attempts": 0, "error": None},
        "feature_engineering": {"success": False, "attempts": 0, "error": None},
        "overall_success": False
    }
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING STOCK DATA PIPELINE")
    logger.info(f"Timestamp: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # ========== STEP 1: Data Gathering ==========
        logger.info("STEP 1/2: Stock Data Gathering")
        gathering_success, gathering_output = run_with_retry(
            script_path=Config.STOCK_DATA_SCRIPT,
            timeout=Config.GATHERING_TIMEOUT,
            script_name="Stock Data Gathering"
        )
        
        status["data_gathering"]["success"] = gathering_success
        status["data_gathering"]["attempts"] = Config.MAX_RETRIES if not gathering_success else 1
        
        if not gathering_success:
            status["data_gathering"]["error"] = gathering_output
            logger.error("Pipeline aborted: Data gathering failed")
            return status
        
        logger.info("")
        logger.info("-" * 80)
        logger.info("")
        
        # ========== STEP 2: Feature Engineering ==========
        logger.info("STEP 2/2: Feature Engineering")
        vector_success, vector_output = run_with_retry(
            script_path=Config.STOCK_VECTOR_SCRIPT,
            timeout=Config.VECTOR_TIMEOUT,
            script_name="Feature Engineering"
        )
        
        status["feature_engineering"]["success"] = vector_success
        status["feature_engineering"]["attempts"] = Config.MAX_RETRIES if not vector_success else 1
        
        if not vector_success:
            status["feature_engineering"]["error"] = vector_output
            logger.error("Pipeline completed with errors: Feature engineering failed")
            return status
        
        # ========== Success ==========
        status["overall_success"] = True
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        status["overall_success"] = False
    
    finally:
        # Calculate duration
        pipeline_end = datetime.now()
        duration = (pipeline_end - pipeline_start).total_seconds()
        status["end_time"] = pipeline_end.isoformat()
        status["duration_seconds"] = duration
        
        logger.info(f"Total execution time: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info("")
        
        # Save status
        save_status(status)
        
        return status


# ============================================================================
# SCHEDULER FUNCTIONS
# ============================================================================

def scheduled_job():
    """Wrapper function for scheduled execution"""
    logger.info("")
    logger.info("🔔 SCHEDULED JOB TRIGGERED")
    run_pipeline()


def run_scheduler():
    """
    Main scheduler loop - runs forever and executes pipeline daily
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("STOCK DATA PIPELINE SCHEDULER")
    logger.info("=" * 80)
    logger.info(f"Schedule: Daily at {Config.SCHEDULE_TIME}")
    logger.info(f"Max retries per script: {Config.MAX_RETRIES}")
    logger.info(f"Retry delay: {Config.RETRY_DELAY_SECONDS}s")
    logger.info(f"Log file: {Config.LOG_FILE}")
    logger.info(f"Status file: {Config.STATUS_FILE}")
    logger.info("=" * 80)
    logger.info("")
    
    # Load last status
    last_status = load_last_status()
    if last_status:
        logger.info("Last pipeline execution:")
        logger.info(f"  Time: {last_status.get('start_time', 'Unknown')}")
        logger.info(f"  Success: {last_status.get('overall_success', False)}")
        logger.info(f"  Duration: {last_status.get('duration_seconds', 0)/60:.1f} min")
        logger.info("")
    
    # Schedule the job
    schedule.every().day.at(Config.SCHEDULE_TIME).do(scheduled_job)
    
    logger.info(f"⏰ Next run scheduled for: {schedule.next_run()}")
    logger.info("")
    logger.info("Press Ctrl+C to stop the scheduler")
    logger.info("")
    
    # Main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("SCHEDULER STOPPED BY USER")
        logger.info("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Scheduler crashed: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Stock Data Pipeline Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduler.py                    # Run scheduler (default)
  python scheduler.py --now              # Run pipeline immediately
  python scheduler.py --status           # Show last execution status
  python scheduler.py --schedule 03:30   # Run daily at 3:30 AM
        """
    )
    
    parser.add_argument(
        '--now',
        action='store_true',
        help='Run pipeline immediately and exit'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show last pipeline execution status and exit'
    )
    
    parser.add_argument(
        '--schedule',
        type=str,
        metavar='HH:MM',
        help=f'Set daily run time (default: {Config.SCHEDULE_TIME})'
    )
    
    args = parser.parse_args()
    
    # Handle --status
    if args.status:
        status = load_last_status()
        if status:
            print("\n" + "=" * 80)
            print("LAST PIPELINE EXECUTION STATUS")
            print("=" * 80)
            print(json.dumps(status, indent=2))
            print("=" * 80 + "\n")
        else:
            print("\nNo status file found. Pipeline has not run yet.\n")
        sys.exit(0)
    
    # Handle --schedule
    if args.schedule:
        try:
            # Validate time format
            datetime.strptime(args.schedule, '%H:%M')
            Config.SCHEDULE_TIME = args.schedule
            logger.info(f"Schedule time set to: {args.schedule}")
        except ValueError:
            logger.error(f"Invalid time format: {args.schedule}. Use HH:MM (e.g., 03:30)")
            sys.exit(1)
    
    # Handle --now
    if args.now:
        logger.info("Running pipeline immediately (--now flag)")
        status = run_pipeline()
        sys.exit(0 if status.get("overall_success") else 1)
    
    # Default: Run scheduler
    run_scheduler()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()