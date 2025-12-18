# stock_vector.py
import pandas as pd
import numpy as np
import json
import math

def load_stock_data(csv_path):
    """Load stock data from CSV file"""
    print(f"Loading stock data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} stocks")
    return df

def parse_market_cap(value):
    """Parse market cap value (handles T, B, M, K suffixes)"""
    if pd.isna(value) or value == '':
        return 0
    
    try:
        value_str = str(value).strip().upper()
        
        # Remove currency symbols
        value_str = value_str.replace('₹', '').replace('$', '').replace(',', '').strip()
        
        # Extract multiplier
        multiplier = 1
        if 'T' in value_str:
            multiplier = 1_000_000_000_000
            value_str = value_str.replace('T', '')
        elif 'B' in value_str:
            multiplier = 1_000_000_000
            value_str = value_str.replace('B', '')
        elif 'M' in value_str:
            multiplier = 1_000_000
            value_str = value_str.replace('M', '')
        elif 'K' in value_str:
            multiplier = 1_000
            value_str = value_str.replace('K', '')
        
        return float(value_str) * multiplier
    except:
        return 0

def parse_percentage(value):
    """Convert percentage string to float (5.23% -> 0.0523)"""
    if pd.isna(value) or value == '':
        return 0
    try:
        value_str = str(value).strip().replace('%', '')
        return float(value_str) / 100.0
    except:
        return 0

def parse_float(value):
    """Safely parse float value"""
    if pd.isna(value) or value == '':
        return 0
    try:
        return float(str(value).replace(',', ''))
    except:
        return 0

def get_market_cap_class(market_cap):
    """
    Determine market cap class from numeric value
    Large Cap: > 20,000 Cr
    Mid Cap: 5,000 - 20,000 Cr
    Small Cap: < 5,000 Cr
    """
    market_cap_cr = market_cap / 10_000_000  # Convert to Crores
    
    if market_cap_cr >= 20000:
        return 'Large', 1.0
    elif market_cap_cr >= 5000:
        return 'Mid', 0.6
    else:
        return 'Small', 0.2

def map_market_cap_preference(market_cap_class_str):
    """Map market cap class string to numeric preference"""
    if pd.isna(market_cap_class_str):
        return 0.5
    
    mc_str = str(market_cap_class_str).strip().lower()
    
    if 'large' in mc_str or 'mega' in mc_str:
        return 1.0
    elif 'mid' in mc_str or 'medium' in mc_str:
        return 0.6
    elif 'small' in mc_str or 'micro' in mc_str:
        return 0.2
    else:
        return 0.5

def robust_normalize(series, method='minmax', clip=True):
    """
    Robust normalization with outlier handling
    """
    values = series.fillna(0).values
    
    if len(values) == 0 or np.all(values == 0):
        return np.zeros_like(values)
    
    if method == 'minmax':
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return np.full_like(values, 0.5)
        normalized = (values - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return np.full_like(values, 0.5)
        normalized = (values - mean_val) / std_val
        # Convert to 0-1 using sigmoid
        normalized = 1 / (1 + np.exp(-normalized))
    
    elif method == 'robust':
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        if iqr == 0:
            return np.full_like(values, 0.5)
        normalized = (values - q25) / iqr
    
    if clip:
        normalized = np.clip(normalized, 0, 1)
    
    return normalized

def clean_stock_data(df):
    """Clean and standardize stock data"""
    print("\n=== Cleaning Stock Data ===")
    
    df_clean = df.copy()
    
    # Parse numeric columns
    if 'market_cap' in df_clean.columns:
        df_clean['market_cap_numeric'] = df_clean['market_cap'].apply(parse_market_cap)
    else:
        df_clean['market_cap_numeric'] = 0
    
    # Parse percentage returns
    return_cols = ['return_1m', 'return_3m', 'return_6m', 'return_1y', 'return_3y']
    for col in return_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_clean'] = df_clean[col].apply(parse_percentage)
        else:
            df_clean[f'{col}_clean'] = 0
    
    # Parse volatility
    vol_cols = ['volatility_30d', 'volatility_90d']
    for col in vol_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_clean'] = df_clean[col].apply(parse_percentage)
        else:
            df_clean[f'{col}_clean'] = 0
    
    # Parse other numeric fields
    numeric_cols = ['beta', 'current_price', 'dividend_yield', 'volume', 
                    'open', 'high', 'low', 'close']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_clean'] = df_clean[col].apply(parse_float)
        else:
            df_clean[f'{col}_clean'] = 0
    
    # Market cap preference
    if 'market_cap_class' in df_clean.columns:
        df_clean['market_cap_pref'] = df_clean['market_cap_class'].apply(map_market_cap_preference)
    else:
        # Infer from numeric market cap
        df_clean['market_cap_pref'] = df_clean['market_cap_numeric'].apply(
            lambda x: get_market_cap_class(x)[1]
        )
    
    print(f"✓ Cleaned {len(df_clean)} stocks")
    return df_clean

def engineer_stock_features(df_clean):
    """
    Engineer 7-dimensional feature vector for each stock
    """
    print("\n=== Engineering Stock Features ===")
    
    df_eng = df_clean.copy()
    
    # Normalize key metrics
    norm_beta = robust_normalize(df_eng['beta_clean'].abs(), method='robust')
    norm_vol_30d = robust_normalize(df_eng['volatility_30d_clean'], method='robust')
    norm_vol_90d = robust_normalize(df_eng['volatility_90d_clean'], method='robust')
    norm_return_1m = robust_normalize(df_eng['return_1m_clean'], method='robust')
    norm_return_3m = robust_normalize(df_eng['return_3m_clean'], method='robust')
    norm_return_6m = robust_normalize(df_eng['return_6m_clean'], method='robust')
    norm_return_1y = robust_normalize(df_eng['return_1y_clean'], method='robust')
    norm_return_3y = robust_normalize(df_eng['return_3y_clean'], method='robust')
    norm_dividend = robust_normalize(df_eng['dividend_yield_clean'], method='minmax')
    
    # 1. RISK SCORE (Higher beta + higher volatility = higher risk)
    df_eng['risk_score'] = (
        0.40 * norm_beta +
        0.35 * norm_vol_30d +
        0.25 * norm_vol_90d
    )
    
    # 2. RETURN SCORE (Long-term weighted returns)
    df_eng['return_score'] = (
        0.30 * norm_return_1y +
        0.40 * norm_return_3y +
        0.20 * norm_return_6m +
        0.10 * norm_return_3m
    )
    
    # 3. STABILITY SCORE (Inverse of volatility + low beta deviation)
    beta_stability = 1 - np.abs(df_eng['beta_clean'] - 1.0) / 2  # Closer to 1 = more stable
    beta_stability = np.clip(beta_stability, 0, 1)
    
    df_eng['stability_score'] = (
        0.40 * (1 - norm_vol_30d) +
        0.35 * (1 - norm_vol_90d) +
        0.25 * beta_stability
    )
    
    # 4. VOLATILITY SCORE (Direct volatility measure)
    df_eng['volatility_score'] = (
        0.60 * norm_vol_30d +
        0.40 * norm_vol_90d
    )
    
    # 5. MARKET CAP PREFERENCE (Already computed)
    df_eng['market_cap_preference'] = df_eng['market_cap_pref']
    
    # 6. DIVIDEND SCORE (Normalized dividend yield)
    df_eng['dividend_score'] = norm_dividend
    
    # 7. MOMENTUM SCORE (Short-term trend strength)
    # Calculate momentum with acceleration
    recent_momentum = norm_return_1m
    medium_momentum = norm_return_3m
    long_momentum = norm_return_6m
    
    # Check for positive acceleration
    acceleration = np.where(
        (recent_momentum > medium_momentum) & (medium_momentum > long_momentum),
        1.2,  # Strong positive acceleration
        np.where(
            recent_momentum > medium_momentum,
            1.1,  # Moderate positive acceleration
            0.9   # No or negative acceleration
        )
    )
    
    df_eng['momentum_score'] = (
        0.40 * recent_momentum +
        0.35 * medium_momentum +
        0.25 * long_momentum
    ) * acceleration
    
    # Clip all scores to [0, 1]
    score_cols = ['risk_score', 'return_score', 'stability_score', 'volatility_score',
                  'market_cap_preference', 'dividend_score', 'momentum_score']
    
    for col in score_cols:
        df_eng[col] = np.clip(df_eng[col], 0, 1)
    
    print(f"✓ Engineered features for {len(df_eng)} stocks")
    return df_eng

def create_output_json(df_eng):
    """Convert engineered DataFrame to JSON output format"""
    print("\n=== Creating JSON Output ===")
    
    results = []
    
    for idx, row in df_eng.iterrows():
        stock_json = {
            "symbol": str(row.get('symbol', row.get('ticker', 'UNKNOWN'))),
            "company_name": str(row.get('company_name', 'Unknown Company')),
            "engineered_vector": [
                round(float(row['risk_score']), 6),
                round(float(row['return_score']), 6),
                round(float(row['stability_score']), 6),
                round(float(row['volatility_score']), 6),
                round(float(row['market_cap_preference']), 6),
                round(float(row['dividend_score']), 6),
                round(float(row['momentum_score']), 6)
            ],
            "metadata": {
                "sector": str(row.get('sector', 'Unknown')),
                "market_cap": round(float(row['market_cap_numeric']), 2),
                "beta": round(float(row['beta_clean']), 2),
                "volatility_30d": round(float(row['volatility_30d_clean']), 4),
                "volatility_90d": round(float(row['volatility_90d_clean']), 4),
                "return_1y": round(float(row['return_1y_clean']), 4),
                "dividend_yield": round(float(row['dividend_yield_clean']), 4),
                "risk_bucket": str(row.get('risk_bucket', 'Medium'))
            },
            "notes": "Vector ready for similarity search (user → stock)"
        }
        results.append(stock_json)
    
    print(f"✓ Created JSON for {len(results)} stocks")
    return results

def process_stocks(csv_path, output_path='engineered_stocks.json'):
    """
    Complete stock feature engineering pipeline
    """
    print("="*60)
    print("STOCK FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    try:
        # Load data
        df = load_stock_data(csv_path)
        
        # Clean data
        df_clean = clean_stock_data(df)
        
        # Engineer features
        df_eng = engineer_stock_features(df_clean)
        
        # Create JSON output
        results = create_output_json(df_eng)
        
        # Save to file
        print(f"\n=== Saving to {output_path} ===")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Successfully saved {len(results)} stocks to {output_path}")
        
        # Print sample
        print("\n=== Sample Output ===")
        for i in range(min(3, len(results))):
            print(f"\n{i+1}. Stock: {results[i]['symbol']} - {results[i]['company_name']}")
            print(f"   Vector: {results[i]['engineered_vector']}")
            print(f"   Sector: {results[i]['metadata']['sector']}")
            print(f"   Market Cap: ₹{results[i]['metadata']['market_cap']:,.0f}")
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results
        
    except FileNotFoundError:
        print(f"\n✗ Error: File '{csv_path}' not found!")
        print("Please provide the correct path to your CSV file.")
        return None
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    # # Input and output file paths
    # input_csv = "data/live_nse_stocks_final1.csv"  # Your stock CSV file
    # output_json = "data/engineered_stocks.json"  # Output JSON file


    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data"

    input_csv = DATA_DIR / "live_nse_stocks_final1.csv"
    output_json = DATA_DIR / "engineered_stocks.json"

    
    # Run the pipeline
    results = process_stocks(input_csv, output_json)
    
    if results:
        print(f"\n✓ Total stocks processed: {len(results)}")
        print(f"✓ Output file: {output_json}")
    else:
        print("\n✗ Pipeline failed. Please check the errors above.")