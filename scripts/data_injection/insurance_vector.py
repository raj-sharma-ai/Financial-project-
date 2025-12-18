import json
import numpy as np
import re
from pathlib import Path

def engineer_insurance_features(policy_data):
    """
    Insurance policy ka 7 normalized features (0-1 range mein)
    """
    
    # 1. COVERAGE_COMPREHENSIVENESS (0-1)
    coverage_items = [
        policy_data.get('covers_preexisting', False),
        policy_data.get('maternity_cover', False),
        policy_data.get('critical_illness_cover', False),
        policy_data.get('hospital_cash_benefit', False),
        policy_data.get('daycare_procedures_covered') in [True, 'Yes', 'yes'],
        policy_data.get('opd_cover', False),
        policy_data.get('ambulance_cover') in [True, 'Yes', 'yes'],
        policy_data.get('annual_health_checkup') in [True, 'Yes', 'yes'],
        policy_data.get('no_claim_bonus') in [True, 'Yes', 'yes']
    ]
    coverage_score = sum(1 for x in coverage_items if x) / len(coverage_items)
    
    # 2. PREMIUM_AFFORDABILITY (0-1)
    premium_features_count = sum([
        policy_data.get('maternity_cover', False),
        policy_data.get('critical_illness_cover', False),
        policy_data.get('opd_cover', False),
        policy_data.get('covers_preexisting', False)
    ])
    affordability_score = 1 - (premium_features_count / 4 * 0.7)
    
    # 3. CLAIM_EASE (0-1)
    waiting_days = policy_data.get('waiting_period_general', '30 days')
    try:
        days = int(re.search(r'\d+', str(waiting_days)).group())
        claim_ease = max(0, 1 - (days / 120))
    except:
        claim_ease = 0.5
    
    # 4. RISK_COVERAGE (0-1)
    risk_items = [
        policy_data.get('covers_preexisting', False),
        policy_data.get('critical_illness_cover', False),
        policy_data.get('ambulance_cover') in [True, 'Yes', 'yes'],
        policy_data.get('hospital_cash_benefit', False)
    ]
    risk_coverage = sum(1 for x in risk_items if x) / len(risk_items)
    
    # 5. NETWORK_ACCESSIBILITY (0-1)
    network = str(policy_data.get('network_hospitals', '')).lower()
    if 'wide' in network or 'large' in network or 'extensive' in network:
        network_score = 1.0
    elif 'medium' in network or 'moderate' in network or 'good' in network:
        network_score = 0.65
    elif 'limited' in network or 'small' in network:
        network_score = 0.3
    else:
        network_score = 0.5
    
    # 6. WELLNESS_BENEFITS (0-1)
    wellness_items = [
        policy_data.get('annual_health_checkup') in [True, 'Yes', 'yes'],
        policy_data.get('opd_cover', False)
    ]
    unique_features = policy_data.get('unique_features', [])
    wellness_keywords = ['wellness', 'health', 'preventive', 'chronic', 'care', 'checkup', 'rewards']
    has_wellness_feature = any(
        keyword in str(feature).lower() 
        for feature in unique_features 
        for keyword in wellness_keywords
    )
    wellness_items.append(has_wellness_feature)
    wellness_score = sum(1 for x in wellness_items if x) / len(wellness_items)
    
    # 7. BONUS_RESTORE_BENEFITS (0-1)
    bonus_items = [
        policy_data.get('no_claim_bonus') in [True, 'Yes', 'yes'],
        policy_data.get('automatic_restore') in [True, 'Yes', 'yes']
    ]
    has_reload = any(
        'reload' in str(feature).lower() or 'restore' in str(feature).lower()
        for feature in unique_features
    )
    bonus_items.append(has_reload)
    bonus_score = sum(1 for x in bonus_items if x) / len(bonus_items)
    
    # Final 7 features
    engineered_vector = [
        round(float(np.clip(coverage_score, 0, 1)), 6),
        round(float(np.clip(affordability_score, 0, 1)), 6),
        round(float(np.clip(claim_ease, 0, 1)), 6),
        round(float(np.clip(risk_coverage, 0, 1)), 6),
        round(float(np.clip(network_score, 0, 1)), 6),
        round(float(np.clip(wellness_score, 0, 1)), 6),
        round(float(np.clip(bonus_score, 0, 1)), 6)
    ]
    
    return engineered_vector


def process_insurance_json(input_path, output_path):
    """
    JSON file se saari policies ko process karta hai
    Aur engineered features ke saath nayi file banata hai
    """
    
    print("=" * 70)
    print("INSURANCE FEATURE ENGINEERING - BATCH PROCESSOR")
    print("=" * 70)
    print(f"\n📂 Reading from: {input_path}")
    
    # Read input JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            policies = [data]
        elif isinstance(data, list):
            policies = data
        else:
            raise ValueError("JSON should be a list or dict")
            
    except FileNotFoundError:
        print(f"❌ Error: File not found - {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return
    
    print(f"✅ Found {len(policies)} policies to process\n")
    
    # Process each policy
    engineered_policies = []
    
    print("🔧 Processing policies...")
    print("-" * 70)
    
    for i, policy in enumerate(policies, 1):
        try:
            # Generate engineered vector
            engineered_vector = engineer_insurance_features(policy)
            
            # Create new policy object with engineered features
            engineered_policy = {
                **policy,  # Keep all original fields
                "engineered_vector": engineered_vector,
                "features": {
                    "coverage_comprehensiveness": engineered_vector[0],
                    "premium_affordability": engineered_vector[1],
                    "claim_ease": engineered_vector[2],
                    "risk_coverage": engineered_vector[3],
                    "network_accessibility": engineered_vector[4],
                    "wellness_benefits": engineered_vector[5],
                    "bonus_restore_benefits": engineered_vector[6]
                }
            }
            
            engineered_policies.append(engineered_policy)
            
            policy_name = policy.get('policy_name', f'Policy_{i}')
            print(f"✓ [{i:3d}] {policy_name:.<45} {engineered_vector}")
            
        except Exception as e:
            policy_name = policy.get('policy_name', f'Policy_{i}')
            print(f"✗ [{i:3d}] {policy_name:.<45} ERROR: {str(e)}")
    
    # Save to output file
    print("\n" + "-" * 70)
    print(f"💾 Saving to: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(engineered_policies, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully saved {len(engineered_policies)} policies!")
        
        # Show summary
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)
        print(f"Total policies processed: {len(engineered_policies)}")
        print(f"Output file size: {Path(output_path).stat().st_size / 1024:.2f} KB")
        print(f"\nFeatures added to each policy:")
        print("  - engineered_vector: [7 features in 0-1 range]")
        print("  - features: {detailed feature breakdown}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
    
    print("\n" + "=" * 70)


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # INPUT/OUTPUT PATHS - Yahan apni file paths daal do
    INPUT_JSON = "health_policies.json"        # Your input file
    OUTPUT_JSON = "engineered_insurance.json"  # Output file with features
    
    # Process the file
    process_insurance_json(INPUT_JSON, OUTPUT_JSON)
    
    print("\n🎯 USAGE EXAMPLE:")
    print("-" * 70)
    print("# Step 1: Iss script ko run karo")
    print("python insurance_feature_engineering.py")
    print()
    print("# Step 2: Input JSON format (list of policies):")
    print("""[
  {
    "policy_name": "Activ One",
    "insurer": "Aditya Birla Capital",
    "waiting_period_general": "30 days",
    "covers_preexisting": true,
    ...
  },
  {
    "policy_name": "Health Shield",
    ...
  }
]""")
    print()
    print("# Step 3: Output mein engineered_vector mil jayega!")
    print("""[
  {
    "policy_name": "Activ One",
    "insurer": "Aditya Birla Capital",
    ...
    "engineered_vector": [0.888889, 0.3, 0.75, 1.0, 1.0, 1.0, 0.666667],
    "features": {
      "coverage_comprehensiveness": 0.888889,
      "premium_affordability": 0.3,
      ...
    }
  }
]""")
    print("\n" + "=" * 70)