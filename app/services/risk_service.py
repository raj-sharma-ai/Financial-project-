"""
Risk Assessment Service
"""
import pandas as pd
from typing import Dict, Optional
from app.core.constants import RISK_SCORE_MAP


def predict_user_risk(user_data: Dict, model_data: Optional[Dict]) -> Dict:
    """Predict risk for a single user"""
    if model_data is None:
        # Fallback prediction based on credit score
        if user_data['creditscore'] >= 750:
            risk_label = 'low'
            risk_score = 0.3
        elif user_data['creditscore'] >= 650:
            risk_label = 'medium'
            risk_score = 0.6
        else:
            risk_label = 'high'
            risk_score = 0.9
        
        return {
            'risk_label': risk_label,
            'risk_score': risk_score
        }
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    user_df = pd.DataFrame([user_data])
    
    # Encode categorical features that exist in the model
    for col in ['gender', 'occupation', 'pastinvestments']:
        if col in user_df.columns and col in label_encoders:
            col_values = user_df[col].astype(str)
            known_classes = set(label_encoders[col].classes_)
            col_values = col_values.apply(lambda x: x if x in known_classes else 'Unknown')
            user_df[f'{col}_encoded'] = label_encoders[col].transform(col_values)
    
    # Create a dictionary with all available features
    available_features = {}
    
    for feature in feature_columns:
        if feature in user_df.columns:
            available_features[feature] = user_df[feature].iloc[0]
        elif feature.endswith('_encoded'):
            if feature in user_df.columns:
                available_features[feature] = user_df[feature].iloc[0]
            else:
                available_features[feature] = 0
        else:
            if 'size' in feature.lower() or 'num' in feature.lower():
                available_features[feature] = 0
            else:
                available_features[feature] = user_df.get(feature, pd.Series([0])).iloc[0] if feature in user_df.columns else 0
    
    # Create DataFrame with all required features
    X = pd.DataFrame([available_features])
    
    # Ensure all feature_columns are present
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Select features in the correct order
    X = X[feature_columns]
    
    # Fill any remaining NaN values
    X = X.fillna(X.median())
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    risk_label = label_encoders['target'].inverse_transform([prediction])[0]
    
    risk_score = RISK_SCORE_MAP.get(risk_label, 0.5)
    
    return {
        'risk_label': risk_label,
        'risk_score': risk_score
    }

