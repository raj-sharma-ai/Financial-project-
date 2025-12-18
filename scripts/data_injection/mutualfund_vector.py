import json
import math
import numpy as np

def load_funds_data(json_path):
    """Load mutual fund data from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def parse_percentage(val):
    """Convert '3.10%' to 0.0310"""
    if not val or val == "" or val == "-":
        return None
    try:
        return float(str(val).strip().strip('%')) / 100.0
    except:
        return None

def parse_float(val):
    """Convert string to float, handle commas"""
    if not val or val == "" or val == "-":
        return None
    try:
        clean_val = str(val).replace(',', '')
        return float(clean_val)
    except:
        return None

def robust_normalize(values, method='minmax'):
    """
    Robust normalization with outlier handling
    method: 'minmax', 'zscore', or 'robust'
    """
    valid_vals = [v for v in values if v is not None]
    if not valid_vals or len(valid_vals) < 2:
        return [0.5 if v is not None else 0 for v in values]
    
    valid_vals_arr = np.array(valid_vals)
    
    if method == 'minmax':
        min_val = np.min(valid_vals_arr)
        max_val = np.max(valid_vals_arr)
        if max_val == min_val:
            return [0.5 if v is not None else 0 for v in values]
        return [(v - min_val) / (max_val - min_val) if v is not None else 0 for v in values]
    
    elif method == 'zscore':
        mean_val = np.mean(valid_vals_arr)
        std_val = np.std(valid_vals_arr)
        if std_val == 0:
            return [0.5 if v is not None else 0 for v in values]
        normalized = [(v - mean_val) / std_val if v is not None else 0 for v in values]
        # Convert to 0-1 range using sigmoid
        return [1 / (1 + math.exp(-x)) for x in normalized]
    
    elif method == 'robust':
        # Use percentile-based normalization (IQR method)
        q25 = np.percentile(valid_vals_arr, 25)
        q75 = np.percentile(valid_vals_arr, 75)
        iqr = q75 - q25
        if iqr == 0:
            return [0.5 if v is not None else 0 for v in values]
        # Clip outliers
        normalized = []
        for v in values:
            if v is None:
                normalized.append(0)
            else:
                norm_v = (v - q25) / iqr
                norm_v = max(0, min(1, norm_v))  # Clip to [0,1]
                normalized.append(norm_v)
        return normalized

def infer_market_cap_from_metrics(aum, std_dev, beta, return_3y):
    """
    Infer market cap category from actual fund metrics when name doesn't specify
    
    Large Cap: Higher AUM, Lower Volatility, Beta close to 1
    Mid Cap: Medium AUM, Medium Volatility, Beta 0.9-1.1
    Small Cap: Lower AUM, Higher Volatility, Beta can vary more
    """
    score = 0
    
    # AUM Analysis (in Crores)
    if aum is not None:
        if aum > 5000:
            score += 3  # Large cap
        elif aum > 1000:
            score += 2  # Mid cap
        elif aum > 100:
            score += 1.5
        else:
            score += 0.5  # Small cap
    
    # Volatility Analysis
    if std_dev is not None:
        if std_dev < 12:
            score += 2  # Large cap (lower volatility)
        elif std_dev < 15:
            score += 1  # Mid cap
        else:
            score += 0.3  # Small cap (higher volatility)
    
    # Beta Analysis
    if beta is not None:
        if 0.85 <= beta <= 1.05:
            score += 2  # Large cap (closer to market)
        elif 0.75 <= beta <= 1.15:
            score += 1  # Mid cap
        else:
            score += 0.5  # Small cap
    
    # 3-year return pattern
    if return_3y is not None:
        if return_3y > 0.25:  # 25%+ returns
            score += 0.5  # Could be small/mid
        elif return_3y > 0.15:
            score += 1
    
    # Convert score to preference (0-7.5 range -> 0.2-1.0)
    max_score = 7.5
    if score >= 5.5:
        return 1.0  # Large Cap
    elif score >= 3.5:
        return 0.65  # Mid Cap
    elif score >= 2:
        return 0.35  # Small-Mid Cap
    else:
        return 0.2  # Small Cap

def get_market_cap_preference(fund_name, aum, std_dev, beta, return_3y):
    """
    Determine market cap with strict rules:
    1. First check fund name
    2. If not found, use actual metrics
    """
    name_lower = fund_name.lower()
    
    # Strict name-based detection
    if any(x in name_lower for x in ['large cap', 'top 100', 'top 50', 'bluechip', 'nifty 50']):
        return 1.0
    elif any(x in name_lower for x in ['small cap', 'smallcap', 'small-cap']):
        return 0.2
    elif any(x in name_lower for x in ['mid cap', 'midcap', 'mid-cap']):
        return 0.5
    elif any(x in name_lower for x in ['large & mid', 'large and mid', 'large-mid']):
        return 0.75
    elif any(x in name_lower for x in ['flexi cap', 'flexicap', 'multi cap', 'multicap']):
        return 0.6
    
    # If name doesn't specify, use metrics
    return infer_market_cap_from_metrics(aum, std_dev, beta, return_3y)

def calculate_sharpe_ratio(returns, std_dev, risk_free_rate=0.06):
    """Calculate Sharpe ratio if missing"""
    if returns is None or std_dev is None or std_dev == 0:
        return 0
    return (returns - risk_free_rate) / std_dev

def calculate_sortino_ratio(return_1y, downside_risk):
    """Sortino ratio - focuses on downside volatility"""
    if downside_risk == 0:
        return 0
    return (return_1y - 0.06) / downside_risk

def calculate_information_ratio(return_3y, tracking_error):
    """Information ratio - risk-adjusted excess return"""
    if tracking_error == 0:
        return 0
    benchmark_return = 0.15  # Assume 15% benchmark
    return (return_3y - benchmark_return) / tracking_error

# Main Processing Pipeline
def process_funds(json_path, output_path='engineered_funds.json'):
    """
    Complete feature engineering pipeline
    """
    print(f"Loading data from: {json_path}")
    funds_data = load_funds_data(json_path)
    print(f"Loaded {len(funds_data)} funds")
    
    # PART A: Data Cleaning
    print("\n=== PART A: Cleaning Data ===")
    cleaned_funds = []
    
    for fund in funds_data:
        cleaned = {
            'fund_name': fund['fund_name'],
            'fund_link': fund.get('fund_link', ''),
            'aum': parse_float(fund['aum']),
            'return_1m': parse_percentage(fund['return_1m']),
            'return_6m': parse_percentage(fund['return_6m']),
            'return_1y': parse_percentage(fund['return_1y']),
            'return_3y': parse_percentage(fund['return_3y']),
            'return_5y': parse_percentage(fund['return_5y']),
            'expense_ratio': parse_percentage(fund['expense_ratio']),
            'sharpe_ratio': parse_float(fund['sharpe_ratio']),
            'standard_deviation': parse_float(fund['standard_deviation']),
            'beta': parse_float(fund['beta']),
            'std_dev_category_avg': parse_float(fund.get('std_dev_category_avg', '')),
            'beta_category_avg': parse_float(fund.get('beta_category_avg', ''))
        }
        cleaned_funds.append(cleaned)
    
    # Fill missing values intelligently
    print("Filling missing values...")
    for fund in cleaned_funds:
        if fund['standard_deviation'] is None:
            fund['standard_deviation'] = fund['std_dev_category_avg'] if fund['std_dev_category_avg'] else 13.0
        if fund['beta'] is None:
            fund['beta'] = fund['beta_category_avg'] if fund['beta_category_avg'] else 1.0
        if fund['expense_ratio'] is None:
            fund['expense_ratio'] = 0.01
        if fund['return_1m'] is None:
            fund['return_1m'] = 0.01
        if fund['return_6m'] is None:
            fund['return_6m'] = fund['return_1y'] * 0.5 if fund['return_1y'] else 0.05
        if fund['return_1y'] is None:
            fund['return_1y'] = fund['return_3y'] * 0.33 if fund['return_3y'] else 0.10
        if fund['return_3y'] is None:
            fund['return_3y'] = fund['return_1y'] * 3 if fund['return_1y'] else 0.15
        if fund['aum'] is None:
            fund['aum'] = 100.0
        if fund['sharpe_ratio'] is None:
            fund['sharpe_ratio'] = calculate_sharpe_ratio(
                fund['return_3y'], 
                fund['standard_deviation']
            )
    
    # PART B: Feature Engineering with Real Math
    print("\n=== PART B: Engineering Features ===")
    
    # Extract arrays for normalization
    std_devs = [f['standard_deviation'] for f in cleaned_funds]
    betas = [f['beta'] for f in cleaned_funds]
    aums = [f['aum'] for f in cleaned_funds]
    expense_ratios = [f['expense_ratio'] for f in cleaned_funds]
    sharpe_ratios = [f['sharpe_ratio'] for f in cleaned_funds]
    returns_1y = [f['return_1y'] for f in cleaned_funds]
    returns_3y = [f['return_3y'] for f in cleaned_funds]
    
    # Robust normalization
    norm_std_devs = robust_normalize(std_devs, method='robust')
    norm_betas = robust_normalize([abs(b - 1.0) for b in betas], method='minmax')  # Distance from market
    norm_aums = robust_normalize([math.log10(a + 1) for a in aums], method='minmax')  # Log scale for AUM
    norm_expense_ratios = robust_normalize(expense_ratios, method='minmax')
    norm_sharpe = robust_normalize(sharpe_ratios, method='zscore')
    norm_returns_1y = robust_normalize(returns_1y, method='robust')
    norm_returns_3y = robust_normalize(returns_3y, method='robust')
    
    # Generate engineered features
    results = []
    
    for i, fund in enumerate(cleaned_funds):
        # 1. RISK SCORE (Multi-factor risk assessment)
        volatility_score = norm_std_devs[i]
        beta_risk = norm_betas[i]  # How far from market
        
        # Downside risk estimation
        downside_risk = fund['standard_deviation'] * math.sqrt(
            max(0, 1 - fund['sharpe_ratio'] / 2)
        ) if fund['sharpe_ratio'] > 0 else fund['standard_deviation']
        
        risk_score = (
            0.35 * volatility_score +           # Volatility weight
            0.25 * beta_risk +                  # Beta deviation from market
            0.20 * (1 - norm_sharpe[i]) +       # Inverse Sharpe (higher Sharpe = lower risk)
            0.20 * norm_expense_ratios[i]       # Higher expense = higher cost risk
        )
        
        # 2. RETURN SCORE (Time-weighted momentum)
        # Recent returns get exponentially higher weight
        return_score = (
            0.15 * fund['return_1m'] * 12 +         # Annualized 1m
            0.25 * fund['return_6m'] * 2 +          # Annualized 6m
            0.30 * fund['return_1y'] +              # 1y return
            0.30 * fund['return_3y']                # 3y return (stable indicator)
        )
        
        # 3. STABILITY SCORE (Consistency + Size)
        consistency = 1 / (1 + fund['standard_deviation'])  # Lower SD = higher consistency
        size_stability = norm_aums[i]  # Larger AUM = more stable
        
        # Coefficient of variation (CV) - return volatility
        cv = fund['standard_deviation'] / fund['return_3y'] if fund['return_3y'] > 0 else 1
        cv_normalized = 1 / (1 + cv)  # Lower CV = better
        
        stability_score = (
            0.40 * consistency +
            0.35 * size_stability +
            0.25 * cv_normalized
        )
        
        # 4. VOLATILITY SCORE (Direct measure)
        volatility_score = norm_std_devs[i]
        
        # 5. MARKET CAP PREFERENCE (Intelligent inference)
        market_cap_preference = get_market_cap_preference(
            fund['fund_name'],
            fund['aum'],
            fund['standard_deviation'],
            fund['beta'],
            fund['return_3y']
        )
        
        # 6. DIVIDEND/VALUE SCORE (Cost efficiency + Returns)
        # Lower expense ratio is better for dividend/value
        cost_efficiency = 1 - norm_expense_ratios[i]
        
        # Adjusted for actual dividend potential
        dividend_score = (
            0.40 * cost_efficiency +
            0.35 * fund['return_1y'] +
            0.25 * (fund['return_3y'] / 3)  # Average annual return
        )
        
        # 7. MOMENTUM SCORE (Trend strength)
        # Calculate momentum with acceleration
        recent_momentum = fund['return_1m'] * 12  # Annualized
        medium_momentum = fund['return_6m'] * 2   # Annualized
        
        # Momentum acceleration (is momentum increasing?)
        if recent_momentum > medium_momentum:
            acceleration = 1.2  # Positive acceleration
        else:
            acceleration = 0.8  # Negative acceleration
        
        momentum_score = (
            0.45 * recent_momentum +
            0.35 * medium_momentum +
            0.20 * fund['return_1y']
        ) * acceleration
        
        # Normalize final scores to [0, 1]
        def clip_normalize(val, min_val=0, max_val=1):
            return max(min_val, min(max_val, val))
        
        result = {
            "fund_name": fund['fund_name'],
            "fund_link": fund['fund_link'],
            "engineered_vector": [
                round(clip_normalize(risk_score), 6),
                round(clip_normalize(return_score, 0, 0.5), 6),
                round(clip_normalize(stability_score), 6),
                round(clip_normalize(volatility_score), 6),
                round(market_cap_preference, 6),
                round(clip_normalize(dividend_score, 0, 0.3), 6),
                round(clip_normalize(momentum_score, -0.2, 0.5), 6)
            ],
            "metadata": {
                "aum": fund['aum'],
                "expense_ratio": round(fund['expense_ratio'], 4),
                "sharpe_ratio": round(fund['sharpe_ratio'], 2),
                "beta": round(fund['beta'], 2),
                "std_dev": round(fund['standard_deviation'], 2)
            },
            "notes": "Vector ready for cosine similarity and ML models"
        }
        results.append(result)
    
    # Save to output file
    print(f"\n=== Saving to {output_path} ===")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Successfully processed {len(results)} funds")
    print(f"✓ Output saved to: {output_path}")
    
    # Print sample
    print("\n=== Sample Output ===")
    for i in range(min(3, len(results))):
        print(f"\nFund: {results[i]['fund_name']}")
        print(f"Vector: {results[i]['engineered_vector']}")
        print(f"Metadata: {results[i]['metadata']}")
    
    return results

# Run the pipeline
if __name__ == "__main__":
    input_file = "data/mutual_funds_20251209_212806.json"  # Your input JSON file path
    output_file = "utils/engineered_funds.json"  # Output file path
    
    try:
        results = process_funds(input_file, output_file)
        print("\n✓ Feature engineering completed successfully!")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{input_file}' not found!")
        print("Please provide the correct path to your JSON file.")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")