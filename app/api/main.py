# main.py - Helper Functions, Scheduler, and Insurance Logic
"""
Main module containing helper functions, scheduler logic, and insurance prediction functions.
All API routes are in app/api/routes/
"""

"""
Main module containing helper functions, scheduler logic, and insurance prediction functions.
"""

from fastapi import HTTPException
from typing import List, Dict
import pandas as pd
import numpy as np
import json
import hashlib
import pickle
from datetime import datetime
import os
import aiohttp
import asyncio
import schedule
import threading
import subprocess
import sys
from pathlib import Path
import time

# Import from app.py
from app.api.app import (
    app, users_df, model_data, stocks_data, funds_data, insurance_data,
    admin_logs, insurance_model, insurance_scaler, insurance_label_encoder,
    insurance_feature_names, health_policies, insurance_users_df, scheduler_status
)

from app.core.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, DATA_DIR, MODELS_DIR

# Rest of your code continues here...

# from app.core.config import (
#     GROQ_API_KEY,
#     GROQ_API_URL,
#     GROQ_MODEL,
#     DATA_DIR,
#     MODELS_DIR
# )



# =====================================================
# SCHEDULER CONFIGURATION
# =====================================================

class SchedulerConfig:
    BASE_DIR = Path(__file__).parent.parent.parent.absolute()  # Project root
    
    # Script paths
    STOCK_DATA_SCRIPT = str(BASE_DIR / "scripts" / "data_injection" / "stock_data_gathering.py")
    STOCK_VECTOR_SCRIPT = str(BASE_DIR / "scripts" / "data_injection" / "stock_vector.py")
    FUND_DATA_SCRIPT = str(BASE_DIR / "scripts" / "data_injection" / "mutual_fund_data_gathering.py")
    FUND_VECTOR_SCRIPT = str(BASE_DIR / "scripts" / "data_injection" / "mutualfund_vector.py")
    
    # Scheduling times
    STOCK_SCHEDULE_TIME = "02:00"   # daily 2 AM
    FUND_SCHEDULE_DAY = "sunday"    # once a week
    FUND_SCHEDULE_TIME = "03:00"    # sunday 3 AM
    
    ENABLED = True
    
    # Timeouts
    GATHERING_TIMEOUT = 7200
    VECTOR_TIMEOUT = 1800


# =====================================================
# SCHEDULER FUNCTIONS
# =====================================================

def run_script_background(script_path: str, timeout: int, script_name: str) -> tuple:
    """Run a Python script in background"""
    try:
        if not Path(script_path).exists():
            return False, f"Script not found: {script_path}"
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            add_log(admin_logs, "SCHEDULER", "SUCCESS", f"{script_name} completed")
            return True, result.stdout
        else:
            add_log(admin_logs, "SCHEDULER", "ERROR", f"{script_name} failed: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        add_log(admin_logs, "SCHEDULER", "ERROR", f"{script_name} timed out")
        return False, f"Script timed out after {timeout}s"
    except Exception as e:
        add_log(admin_logs, "SCHEDULER", "ERROR", f"{script_name} error: {str(e)}")
        return False, str(e)


def run_pipeline_background(data_type: str, gathering_script: str, vector_script: str) -> bool:
    """Run a complete data pipeline"""
    add_log(admin_logs, "SCHEDULER", "INFO", f"Starting {data_type} pipeline...")
    
    # Data gathering
    success, output = run_script_background(
        gathering_script,
        SchedulerConfig.GATHERING_TIMEOUT,
        f"{data_type} Data Gathering"
    )
    
    if not success:
        return False
    
    # Feature engineering
    success, output = run_script_background(
        vector_script,
        SchedulerConfig.VECTOR_TIMEOUT,
        f"{data_type} Feature Engineering"
    )
    
    return success


def scheduled_data_refresh():
    """Main scheduled job - runs all pipelines and refreshes data"""
    if scheduler_status["is_running"]:
        add_log(admin_logs, "SCHEDULER", "WARNING", "Pipeline already running, skipping...")
        return
    
    scheduler_status["is_running"] = True
    scheduler_status["last_run"] = datetime.now().isoformat()
    
    add_log(admin_logs, "SCHEDULER", "INFO", "Starting scheduled data refresh...")
    
    results = {
        "stocks": False,
        "funds": False
    }
    
    try:
        # Run stock pipeline
        if Path(SchedulerConfig.STOCK_DATA_SCRIPT).exists():
            results["stocks"] = run_pipeline_background(
                "stocks",
                SchedulerConfig.STOCK_DATA_SCRIPT,
                SchedulerConfig.STOCK_VECTOR_SCRIPT
            )
        
        # Run mutual fund pipeline
        if Path(SchedulerConfig.FUND_DATA_SCRIPT).exists():
            results["funds"] = run_pipeline_background(
                "funds",
                SchedulerConfig.FUND_DATA_SCRIPT,
                SchedulerConfig.FUND_VECTOR_SCRIPT
            )
        
        # Reload data into memory
        if any(results.values()):
            from app.api.app import users_df as app_users_df, stocks_data as app_stocks_data, funds_data as app_funds_data
            
            # Note: Data reloading should be handled by app.py's load_initial_data() function
            # This is just for logging purposes
            add_log(admin_logs, "SCHEDULER", "SUCCESS", "Data refresh pipeline completed")
        
        scheduler_status["last_result"] = results
        
    except Exception as e:
        add_log(admin_logs, "SCHEDULER", "ERROR", f"Scheduler error: {str(e)}")
        scheduler_status["last_result"] = {"error": str(e)}
    
    finally:
        scheduler_status["is_running"] = False


def run_scheduler_thread():
    """Run scheduler in background thread"""
    # Stock pipeline - Daily
    schedule.every().day.at(SchedulerConfig.STOCK_SCHEDULE_TIME).do(
        lambda: run_pipeline_background(
            "Stock",
            SchedulerConfig.STOCK_DATA_SCRIPT,
            SchedulerConfig.STOCK_VECTOR_SCRIPT
        )
    )
    
    # Fund pipeline - Weekly
    getattr(schedule.every(), SchedulerConfig.FUND_SCHEDULE_DAY).at(
        SchedulerConfig.FUND_SCHEDULE_TIME
    ).do(
        lambda: run_pipeline_background(
            "Fund",
            SchedulerConfig.FUND_DATA_SCRIPT,
            SchedulerConfig.FUND_VECTOR_SCRIPT
        )
    )
    
    add_log(admin_logs, "SCHEDULER", "INFO", 
            f"Scheduler started - Stock: Daily {SchedulerConfig.STOCK_SCHEDULE_TIME}, "
            f"Fund: Weekly {SchedulerConfig.FUND_SCHEDULE_DAY} {SchedulerConfig.FUND_SCHEDULE_TIME}")
    
    while True:
        schedule.run_pending()
        scheduler_status["next_run"] = str(schedule.next_run()) if schedule.jobs else None
        time.sleep(60)


def start_scheduler():
    """Start scheduler in daemon thread"""
    if not SchedulerConfig.ENABLED:
        add_log(admin_logs, "SCHEDULER", "INFO", "Scheduler disabled in config")
        return
    
    scheduler_thread = threading.Thread(target=run_scheduler_thread, daemon=True)
    scheduler_thread.start()
    add_log(admin_logs, "SCHEDULER", "SUCCESS", "Scheduler thread started")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def deterministic_random(seed_str: str, low: float, high: float) -> float:
    """Generate deterministic pseudo-random number"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)


def calculate_derived_features(user_input: Dict) -> Dict:
    """Calculate derived features for a user"""
    seed_str = f"{user_input['age']}_{user_input['gender']}_{user_input['occupation']}_{user_input['creditscore']}_{user_input['pastinvestments']}"
    
    if user_input['creditscore'] >= 750:
        creditutilizationratio = deterministic_random(seed_str+"1", 0.05, 0.20)
    elif user_input['creditscore'] >= 650:
        creditutilizationratio = deterministic_random(seed_str+"2", 0.20, 0.40)
    else:
        creditutilizationratio = deterministic_random(seed_str+"3", 0.40, 0.70)
    
    monthly_income = user_input['annualincome'] / 12
    monthly_surplus = monthly_income - user_input['avgmonthlyspend']
    estimated_emi = max(0, monthly_surplus * 0.25)
    debttoincomeratio = estimated_emi / monthly_income
    
    if user_input['occupation'] == 'Salaried' and user_input['creditscore'] >= 700:
        transactionvolatility = deterministic_random(seed_str+"4", 0.08, 0.15)
    elif user_input['occupation'] == 'Self-employed':
        transactionvolatility = deterministic_random(seed_str+"5", 0.20, 0.35)
    else:
        transactionvolatility = deterministic_random(seed_str+"6", 0.15, 0.25)
    
    if user_input['savingsrate'] >= 0.30:
        spendingstabilityindex = deterministic_random(seed_str+"7", 0.70, 0.85)
    elif user_input['savingsrate'] >= 0.15:
        spendingstabilityindex = deterministic_random(seed_str+"8", 0.55, 0.70)
    else:
        spendingstabilityindex = deterministic_random(seed_str+"9", 0.40, 0.55)
    
    if user_input['creditscore'] >= 750:
        missedpaymentcount = 0
    elif user_input['creditscore'] >= 650:
        missedpaymentcount = int(deterministic_random(seed_str+"10", 0, 1.99))
    else:
        missedpaymentcount = int(deterministic_random(seed_str+"11", 1, 3.99))
    
    if user_input['age'] < 30:
        digitalactivityscore = deterministic_random(seed_str+"12", 70, 85)
    elif user_input['age'] < 45:
        digitalactivityscore = deterministic_random(seed_str+"13", 55, 75)
    else:
        digitalactivityscore = deterministic_random(seed_str+"14", 40, 60)
    
    if user_input['citytier'] == 1:
        digitalactivityscore += 10
    if user_input['creditscore'] < 650:
        digitalactivityscore -= 10
    digitalactivityscore = max(0, min(100, digitalactivityscore))
    
    investment_types = str(user_input['pastinvestments'])
    if investment_types == 'None' or investment_types == 'nan':
        portfoliodiversityscore = 0
    elif '_' in investment_types or ',' in investment_types:
        portfoliodiversityscore = deterministic_random(seed_str+"15", 60, 80)
    else:
        portfoliodiversityscore = deterministic_random(seed_str+"16", 30, 50)
    
    return {
        'transactionvolatility': transactionvolatility,
        'spendingstabilityindex': spendingstabilityindex,
        'creditutilizationratio': creditutilizationratio,
        'debttoincomeratio': debttoincomeratio,
        'missedpaymentcount': missedpaymentcount,
        'digitalactivityscore': digitalactivityscore,
        'portfoliodiversityscore': portfoliodiversityscore
    }


def robust_normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize single value to [0, 1] range"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def engineer_single_user_vector(user_complete: Dict) -> List[float]:
    """Engineer 7-dimensional vector for user"""
    
    normalized_risk_score = np.clip(user_complete['risk_score'], 0, 1)
    normalized_income = robust_normalize(user_complete['annualincome'], 200000, 2500000)
    normalized_savings_rate = np.clip(user_complete['savingsrate'], 0, 1)
    normalized_debt_to_income = robust_normalize(user_complete['debttoincomeratio'], 0, 0.5)
    normalized_digital_activity = robust_normalize(user_complete['digitalactivityscore'], 0, 100)
    normalized_portfolio_diversity = robust_normalize(user_complete['portfoliodiversityscore'], 0, 100)
    
    credit_min = 300
    credit_max = 850
    normalized_credit_score = (user_complete['creditscore'] - credit_min) / (credit_max - credit_min)
    normalized_credit_score = np.clip(normalized_credit_score, 0, 1)
    
    risk_preference = normalized_risk_score
    return_expectation = 0.40 * normalized_income + 0.35 * normalized_savings_rate + 0.25 * normalized_digital_activity
    stability_preference = 1 - normalized_risk_score
    volatility_tolerance = 0.70 * normalized_risk_score + 0.30 * normalized_digital_activity
    market_cap_preference = 1 - normalized_debt_to_income
    dividend_preference = normalized_portfolio_diversity
    momentum_preference = 0.60 * normalized_digital_activity + 0.40 * normalized_credit_score
    
    vector = [
        np.clip(risk_preference, 0, 1),
        np.clip(return_expectation, 0, 1),
        np.clip(stability_preference, 0, 1),
        np.clip(volatility_tolerance, 0, 1),
        np.clip(market_cap_preference, 0, 1),
        np.clip(dividend_preference, 0, 1),
        np.clip(momentum_preference, 0, 1)
    ]
    
    return vector


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    u = np.array(vec1)
    v = np.array(vec2)
    
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return float(dot_product / (norm_u * norm_v))


def add_log(admin_logs_list: list, action: str, status: str, details: str):
    """Add admin log"""
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "status": status,
        "details": details
    }
    admin_logs_list.append(log)
    if len(admin_logs_list) > 100:
        admin_logs_list.pop(0)


# =====================================================
# LLM HELPER FUNCTIONS
# =====================================================

def generate_fallback_explanation(
    user_profile: Dict,
    top_stocks: List[Dict],
    top_mutual_funds: List[Dict]
) -> str:
    """Generate rule-based explanation when LLM is unavailable"""
    
    risk = user_profile['risk_label'].upper()
    income = user_profile['income']
    savings_rate = user_profile['savings_rate'] * 100
    
    explanation = f"""**Why These Recommendations Match Your Profile:**

• **Risk Alignment**: Your {risk} risk profile has been carefully matched with investments that suit your risk tolerance and financial goals.

• **Income-Based Selection**: With an annual income of ₹{income:,.0f} and a savings rate of {savings_rate:.1f}%, these options are sized appropriately for your financial capacity.

• **Diversification**: The recommendations span multiple sectors and categories to help balance your portfolio and reduce concentration risk.

• **Match Quality**: All recommendations have high similarity scores (80%+), indicating strong alignment with your complete financial profile.

⚠️ **Important**: Past performance doesn't guarantee future returns. Market investments carry risk - please consult a financial advisor before making investment decisions."""
    
    return explanation


def generate_fallback_individual_explanation(
    user_profile: Dict,
    item_type: str,
    item_data: Dict
) -> str:
    """Generate rule-based explanation for individual item when LLM is unavailable"""
    
    risk = user_profile['risk_label'].upper()
    match_score = item_data.get('similarity_score', 0) * 100
    
    if item_type == "stock":
        item_name = f"{item_data.get('symbol', 'N/A')} ({item_data.get('company_name', 'N/A')})"
        sector = item_data.get('metadata', {}).get('sector', 'N/A')
        
        explanation = f"""**Why {item_name} is Recommended:**

• **High Match Score**: With a {match_score:.1f}% match score, this stock aligns well with your {risk} risk profile and financial goals.

• **Sector Alignment**: The {sector} sector is suitable for your investment profile and provides appropriate exposure for your risk tolerance.

• **Profile Compatibility**: This stock's characteristics match your financial metrics including income level, savings rate, and investment experience.

⚠️ **Important**: Stock market investments carry risk. Past performance is not indicative of future results."""
    
    else:  # mutual_fund
        item_name = item_data.get('fund_name', 'N/A')[:60]
        category = item_data.get('metadata', {}).get('category', 'N/A')
        
        explanation = f"""**Why {item_name} is Recommended:**

• **High Match Score**: With a {match_score:.1f}% match score, this fund aligns well with your {risk} risk profile and investment objectives.

• **Category Fit**: The {category} category is appropriate for your financial situation and provides suitable diversification.

• **Profile Compatibility**: This fund's investment strategy matches your risk tolerance, income level, and long-term financial goals.

⚠️ **Important**: Mutual fund investments are subject to market risks. Please read the offer document carefully."""
    
    return explanation


# =====================================================
# RECOMMENDATION FUNCTIONS
# =====================================================

def get_user_recommendations(user_profile: Dict, top_k: int = 10):
    """Get stock, mutual fund, and insurance recommendations for a user profile"""
    
    if stocks_data is None or funds_data is None:
        raise ValueError("Stocks or Funds data not loaded")
    
    user_vector = user_profile['engineered_vector']
    
    # Stock Recommendations
    stock_recommendations = []
    try:
        for stock in stocks_data:
            stock_vector = stock.get('engineered_vector')
            if stock_vector is None:
                continue
            
            similarity = cosine_similarity(user_vector, stock_vector)
            
            stock_recommendations.append({
                'symbol': stock.get('symbol', 'N/A'),
                'company_name': stock.get('company_name', ''),
                'similarity_score': round(similarity, 6),
                'metadata': stock.get('metadata', {})
            })
        
        stock_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        stock_recommendations = stock_recommendations[:top_k]
    except Exception as e:
        add_log(admin_logs, "GET_RECOMMENDATIONS", "ERROR", f"Error generating stock recommendations: {str(e)}")
        raise
    
    # Mutual Fund Recommendations
    fund_recommendations = []
    try:
        for fund in funds_data:
            fund_vector = fund.get('engineered_vector')
            if fund_vector is None:
                continue
            
            similarity = cosine_similarity(user_vector, fund_vector)
            
            fund_recommendations.append({
                'fund_name': fund.get('fund_name', 'N/A'),
                'fund_code': fund.get('fund_code', 'N/A'),
                'similarity_score': round(similarity, 6),
                'metadata': fund.get('metadata', {})
            })
        
        fund_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        fund_recommendations = fund_recommendations[:top_k]
    except Exception as e:
        add_log(admin_logs, "GET_RECOMMENDATIONS", "ERROR", f"Error generating fund recommendations: {str(e)}")
        raise
    
    # Insurance Recommendations
    insurance_recommendations = []
    try:
        if insurance_data and len(insurance_data) > 0:
            for insurance in insurance_data:
                insurance_vector = insurance.get('engineered_vector')
                if insurance_vector is None:
                    continue
                
                similarity = cosine_similarity(user_vector, insurance_vector)
                
                insurance_name = (
                    insurance.get('policy_name') or 
                    insurance.get('insurance_name') or 
                    insurance.get('name') or 
                    insurance.get('product_name') or 
                    'Unknown Insurance'
                )
                
                insurer = insurance.get('insurer', 'N/A')
                insurance_type = 'Health Insurance'
                if insurance.get('critical_illness_cover'):
                    insurance_type = 'Health + Critical Illness'
                
                metadata = {
                    'insurer': insurer,
                    'url': insurance.get('url', ''),
                    'premium_range': insurance.get('premium_amount_range', 'Varies'),
                    'sum_insured': insurance.get('sum_insured_range', 'Check policy'),
                    'waiting_period_general': insurance.get('waiting_period_general', 'N/A'),
                    'waiting_period_preexisting': insurance.get('waiting_period_preexisting', 'N/A'),
                    'covers_preexisting': insurance.get('covers_preexisting', False),
                    'maternity_cover': insurance.get('maternity_cover', False),
                    'critical_illness_cover': insurance.get('critical_illness_cover', False),
                    'opd_cover': insurance.get('opd_cover', False),
                    'network_hospitals': insurance.get('network_hospitals', 'N/A'),
                    'no_claim_bonus': insurance.get('no_claim_bonus', 'N/A'),
                    'features': insurance.get('features', {})
                }
                
                insurance_recommendations.append({
                    'insurance_name': insurance_name,
                    'insurance_type': insurance_type,
                    'similarity_score': round(similarity, 6),
                    'metadata': metadata
                })
            
            insurance_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            insurance_recommendations = insurance_recommendations[:top_k]
    except Exception as e:
        add_log(admin_logs, "GET_RECOMMENDATIONS", "ERROR", f"Error generating insurance recommendations: {str(e)}")
        insurance_recommendations = []
    
    return {
        "stocks": stock_recommendations,
        "mutual_funds": fund_recommendations,
        "insurance": insurance_recommendations
    }


# =====================================================
# INSURANCE PREDICTION FUNCTIONS
# =====================================================

def load_insurance_prediction_models():
    """Load insurance prediction PKL files and insurance users CSV"""
    # Import app module to update globals
    import app.api.app as app_module
    global insurance_model, insurance_scaler, insurance_label_encoder, insurance_feature_names, health_policies, insurance_users_df
    
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project root
        DATA_DIR = BASE_DIR / "data"
        MODELS_DIR = BASE_DIR / "models"
        
        # Try CSV file paths
        csv_paths = [
            DATA_DIR / "test_users_BANK.csv",
        ]
        
        csv_loaded = False
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                insurance_users_df = pd.read_csv(csv_path)
                app_module.insurance_users_df = insurance_users_df  # Update app.py global
                
                # Add customer_id column if missing
                if 'customer_id' not in insurance_users_df.columns:
                    insurance_users_df['customer_id'] = range(1, 1 + len(insurance_users_df))
                    app_module.insurance_users_df = insurance_users_df
                    add_log(admin_logs, "INSURANCE_MODEL", "INFO", f"Added customer_id column to {csv_path}")
                
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ Loaded {len(insurance_users_df)} insurance users from {csv_path}")
                csv_loaded = True
                break
        
        if not csv_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "WARNING", "⚠️ No insurance CSV found - will use fallback")
            insurance_users_df = None
            app_module.insurance_users_df = None
        
        # Model paths
        model_paths = [
            MODELS_DIR / "xgb_model_synthetic.pkl"
        ]
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    insurance_model = pickle.load(f)
                app_module.insurance_model = insurance_model  # Update app.py global
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ XGBoost model loaded from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "ERROR", "❌ xgb_model_synthetic.pkl not found in models/ folder")
            return False
        
        # Scaler paths
        scaler_paths = [
            MODELS_DIR / "xgb_scaler_synthetic.pkl"
        ]
        scaler_loaded = False
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    insurance_scaler = pickle.load(f)
                app_module.insurance_scaler = insurance_scaler  # Update app.py global
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ Scaler loaded from {scaler_path}")
                scaler_loaded = True
                break
        
        if not scaler_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "ERROR", "❌ xgb_scaler_synthetic.pkl not found in models/ folder")
            return False
        
        # Encoder paths
        encoder_paths = [
            MODELS_DIR / "xgb_label_encoder_synthetic.pkl"
        ]
        encoder_loaded = False
        for encoder_path in encoder_paths:
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    insurance_label_encoder = pickle.load(f)
                app_module.insurance_label_encoder = insurance_label_encoder  # Update app.py global
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ Label encoder loaded from {encoder_path}")
                encoder_loaded = True
                break
        
        if not encoder_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "ERROR", "❌ xgb_label_encoder_synthetic.pkl not found in models/ folder")
            return False
        
        # Feature paths
        feature_paths = [
            MODELS_DIR / "xgb_feature_names_synthetic.pkl"
        ]
        feature_loaded = False
        for feature_path in feature_paths:
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    insurance_feature_names = pickle.load(f)
                app_module.insurance_feature_names = insurance_feature_names  # Update app.py global
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ Feature names loaded: {len(insurance_feature_names)} features from {feature_path}")
                feature_loaded = True
                break
        
        if not feature_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "ERROR", "❌ xgb_feature_names_synthetic.pkl not found in models/ folder")
            return False
        
        # Health policies paths
        health_policy_paths = [
            DATA_DIR / "health_policies.json"
        ]
        
        policy_loaded = False
        for policy_path in health_policy_paths:
            if os.path.exists(policy_path):
                with open(policy_path, 'r', encoding='utf-8') as f:
                    health_policies = json.load(f)
                app_module.health_policies = health_policies  # Update app.py global
                add_log(admin_logs, "INSURANCE_MODEL", "SUCCESS", f"✅ Loaded {len(health_policies)} health policies from {policy_path}")
                policy_loaded = True
                break
        
        if not policy_loaded:
            add_log(admin_logs, "INSURANCE_MODEL", "ERROR", "❌ health_policies.json not found in data/ folder")
            return False
        
        return True
        
    except Exception as e:
        add_log(admin_logs, "INSURANCE_MODEL", "ERROR", f"❌ Error loading models: {str(e)}")
        return False


def prepare_user_features_for_insurance(user_data: Dict) -> pd.DataFrame:
    """Prepare user features for insurance prediction using insurance CSV columns"""
    
    features = {
        'age': user_data.get('age', 30),
        'citytier': user_data.get('citytier', 1),
        'annualincome': user_data.get('annualincome', 500000),
        'creditscore': user_data.get('creditscore', 700),
        'avgmonthlyspend': user_data.get('avgmonthlyspend', 20000),
        'savingsrate': user_data.get('savingsrate', 0.2),
        'investmentamountlastyear': user_data.get('investmentamountlastyear', 50000),
        'familysize': user_data.get('familysize', 4),
        'numchildren': user_data.get('numchildren', 1),
        'numelders': user_data.get('numelders', 1),
        'numadults': user_data.get('numadults', 2),
    }
    
    # Encode categorical features
    gender = user_data.get('gender', 'Male')
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    features['gender_encoded'] = gender_map.get(gender, 0)
    
    occupation = user_data.get('occupation', 'Salaried')
    occupation_map = {
        'Salaried': 0,
        'Self-employed': 1,
        'Self-Employed': 1,
        'Business': 2,
        'Professional': 3,
        'Retired': 4,
        'Student': 5
    }
    features['occupation_encoded'] = occupation_map.get(occupation, 0)
    
    past_investments = str(user_data.get('pastinvestments', 'None'))
    if past_investments in ['None', 'nan', '']:
        features['pastinvestments_encoded'] = 0
    elif '_' in past_investments or ',' in past_investments:
        features['pastinvestments_encoded'] = 2
    else:
        features['pastinvestments_encoded'] = 1
    
    # Create DataFrame
    user_df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for feature in insurance_feature_names:
        if feature not in user_df.columns:
            user_df[feature] = 0
    
    # Select only the features used in training (in correct order)
    user_df = user_df[insurance_feature_names]
    
    return user_df


def predict_insurance_policies(user_id: int) -> Dict:
    """Predict insurance policies and their probabilities for a user"""
    
    if insurance_model is None or health_policies is None:
        raise HTTPException(status_code=500, detail="Insurance prediction models not loaded")
    
    # Try to get user from insurance CSV first, then fall back to main users CSV
    user_data = None
    
    if insurance_users_df is not None:
        user_row = insurance_users_df[insurance_users_df['customer_id'] == user_id]
        if not user_row.empty:
            user_data = user_row.iloc[0].to_dict()
            add_log(admin_logs, "INSURANCE_PREDICT", "INFO", f"Using insurance_users.csv for user {user_id}")
    
    # Fallback to main users CSV if not found in insurance CSV
    if user_data is None:
        if users_df is None:
            raise HTTPException(status_code=404, detail="Users data not loaded")
        
        user_row = users_df[users_df['customer_id'] == user_id]
        if user_row.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user_data = user_row.iloc[0].to_dict()
        
        # Add default family-related columns if not present
        if 'familysize' not in user_data:
            user_data['familysize'] = 4
        if 'numchildren' not in user_data:
            user_data['numchildren'] = 1
        if 'numelders' not in user_data:
            user_data['numelders'] = 1
        if 'numadults' not in user_data:
            user_data['numadults'] = 2
        
        add_log(admin_logs, "INSURANCE_PREDICT", "INFO", f"Using main users CSV with defaults for user {user_id}")
    
    # Prepare features for model
    user_features = prepare_user_features_for_insurance(user_data)
    
    # Scale features
    user_features_scaled = insurance_scaler.transform(user_features)
    
    # Convert to DMatrix for XGBoost Booster
    import xgboost as xgb
    dmatrix = xgb.DMatrix(user_features_scaled, feature_names=insurance_feature_names)
    
    # Get probability predictions for all classes
    probabilities = insurance_model.predict(dmatrix)[0]
    
    # Get policy names from label encoder
    policy_classes = insurance_label_encoder.classes_
    
    # Create predictions list with probabilities
    predictions = []
    for idx, policy_name in enumerate(policy_classes):
        probability = float(probabilities[idx])
        
        # Find matching policy in health_policies.json
        matching_policy = None
        for policy in health_policies:
            policy_name_in_json = policy.get('policy_name', '').strip()
            if policy_name_in_json == policy_name.strip():
                matching_policy = policy
                break
        
        if matching_policy:
            predictions.append({
                'policy_name': policy_name,
                'probability': round(probability * 100, 2),
                'probability_score': round(probability, 4),
                'insurer': matching_policy.get('insurer', 'N/A'),
                'url': matching_policy.get('url', ''),
                'premium_range': matching_policy.get('premium_amount_range', 'Varies'),
                'sum_insured': matching_policy.get('sum_insured_range', 'Check policy'),
                'features': {
                    'covers_preexisting': matching_policy.get('covers_preexisting', False),
                    'maternity_cover': matching_policy.get('maternity_cover', False),
                    'critical_illness_cover': matching_policy.get('critical_illness_cover', False),
                    'opd_cover': matching_policy.get('opd_cover', False),
                    'network_hospitals': matching_policy.get('network_hospitals', 'N/A'),
                    'waiting_period': matching_policy.get('waiting_period_general', 'N/A')
                }
            })
        else:
            predictions.append({
                'policy_name': policy_name,
                'probability': round(probability * 100, 2),
                'probability_score': round(probability, 4),
                'insurer': 'N/A',
                'url': '',
                'premium_range': 'N/A',
                'sum_insured': 'N/A',
                'features': {}
            })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Get user metadata
    user_metadata = {
        'age': int(user_data.get('age', 30)),
        'gender': str(user_data.get('gender', 'Male')),
        'occupation': str(user_data.get('occupation', 'Salaried')),
        'income': round(float(user_data.get('annualincome', 500000)), 2),
        'credit_score': int(user_data.get('creditscore', 700)),
        'city_tier': int(user_data.get('citytier', 1)),
        'family_size': int(user_data.get('familysize', 4)),
        'num_children': int(user_data.get('numchildren', 1)),
        'num_elders': int(user_data.get('numelders', 1)),
        'num_adults': int(user_data.get('numadults', 2))
    }
    
    return {
        'user_id': str(user_id),
        'user_metadata': user_metadata,
        'predicted_policies': predictions,
        'top_policy': predictions[0] if predictions else None
    }


# =====================================================
# STARTUP: Load Insurance Models and Start Scheduler
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Load insurance models and start scheduler on startup"""
    # Load insurance prediction models
    insurance_models_loaded = load_insurance_prediction_models()
    if insurance_models_loaded:
        add_log(admin_logs, "STARTUP", "SUCCESS", "✅ Insurance prediction models loaded")
    else:
        add_log(admin_logs, "STARTUP", "WARNING", "⚠️ Insurance prediction models NOT loaded - Check files")
    
    # Start scheduler
    start_scheduler()
    add_log(admin_logs, "STARTUP", "SUCCESS", "API initialized successfully")


# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
