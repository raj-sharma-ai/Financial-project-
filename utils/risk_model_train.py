
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Manual SMOTE Implementation
# ------------------------------
class SimpleSMOTE:
    def __init__(self, k_neighbors=5, random_state=None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        # Find minority classes
        unique_classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            cls_count = len(cls_indices)
            
            if cls_count < max_count:
                # Number of synthetic samples needed
                n_synthetic = max_count - cls_count
                
                # Get samples of this class
                X_cls = X[cls_indices]
                
                # Fit nearest neighbors
                k = min(self.k_neighbors, len(X_cls) - 1)
                if k <= 0:
                    continue
                    
                nn = NearestNeighbors(n_neighbors=k + 1)
                nn.fit(X_cls)
                
                # Generate synthetic samples
                for _ in range(n_synthetic):
                    # Random sample from minority class
                    idx = np.random.randint(0, len(X_cls))
                    sample = X_cls[idx]
                    
                    # Find k nearest neighbors
                    neighbors_idx = nn.kneighbors([sample], return_distance=False)[0][1:]
                    
                    # Choose random neighbor
                    nn_idx = np.random.choice(neighbors_idx)
                    neighbor = X_cls[nn_idx]
                    
                    # Generate synthetic sample
                    alpha = np.random.random()
                    synthetic = sample + alpha * (neighbor - sample)
                    
                    # Add to dataset
                    X_resampled = np.vstack([X_resampled, synthetic])
                    y_resampled = np.append(y_resampled, cls)
        
        return X_resampled, y_resampled

# ------------------------------
# Helper for deterministic random
# ------------------------------
def deterministic_random(seed_str, low, high):
    """Generate deterministic pseudo-random number based on string seed"""
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    return np.random.uniform(low, high)

# ------------------------------
# Main Risk Profiling Model Class
# ------------------------------
class RiskProfilingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    # ------------------------------
    # Data Loading and Merging
    # ------------------------------
    def load_data(self, customer_info_path, customer_behavior_path):
        print("Loading data...")
        customer_info = pd.read_csv(customer_info_path)
        customer_behavior = pd.read_csv(customer_behavior_path)
        
        # Standardize column names
        customer_info.columns = customer_info.columns.str.strip().str.lower()
        customer_behavior.columns = customer_behavior.columns.str.strip().str.lower()
        
        # Rename customerid if needed
        if 'customerid' in customer_info.columns:
            customer_info = customer_info.rename(columns={'customerid': 'customer_id'})
        if 'customerid' in customer_behavior.columns:
            customer_behavior = customer_behavior.rename(columns={'customerid': 'customer_id'})
        
        df = customer_info.merge(customer_behavior, on='customer_id', how='inner')
        print(f"✅ Total unique customers: {len(df)}")
        return df
    
    # ------------------------------
    # Risk Label Creation
    # ------------------------------
    def create_risk_labels(self, df):
        credit_health = (
            (df['creditscore'] / 850) * 0.25 +
            (1 - df['creditutilizationratio']) * 0.08 +
            (1 - df['debttoincomeratio']) * 0.07
        )
        
        payment_behavior = (
            (1 - (df['missedpaymentcount'] / df['missedpaymentcount'].max()).fillna(0)) * 0.20 +
            (1 - df['transactionvolatility']) * 0.05
        )
        
        income_stability = (
            (df['annualincome'] / df['annualincome'].max()) * 0.12 +
            df['spendingstabilityindex'] * 0.08
        )
        
        savings_assets = (
            (df['savingsrate'] * 2).clip(0,1) * 0.05 +
            (df['investmentamountlastyear'] / df['investmentamountlastyear'].max()).fillna(0) * 0.05
        )
        
        portfolio = (
            (df['portfoliodiversityscore'] / 100) * 0.03 +
            (df['digitalactivityscore'] / 100) * 0.02
        )
        
        risk_score = (credit_health + payment_behavior + income_stability + savings_assets + portfolio).clip(0,1)
        
        df['risk_score'] = risk_score
        df['risk_label'] = pd.cut(
            risk_score,
            bins=[0,0.55,0.72,1.0],
            labels=['high','medium','low']
        )
        
        print("\n📊 Risk Distribution:")
        print(df['risk_label'].value_counts())
        
        return df
    
    # ------------------------------
    # Feature Preprocessing
    # ------------------------------
    def preprocess_features(self, df, is_training=True):
        categorical_cols = ['gender', 'occupation', 'pastinvestments']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    values = df[col].astype(str).unique().tolist()
                    if "Unknown" not in values:
                        values.append("Unknown")
                    self.label_encoders[col].fit(values)
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    col_values = df[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    col_values = col_values.apply(lambda x: x if x in known_classes else 'Unknown')
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(col_values)
        
        self.feature_columns = [
            'age','citytier','annualincome','creditscore',
            'avgmonthlyspend','savingsrate','investmentamountlastyear',
            'transactionvolatility','spendingstabilityindex',
            'creditutilizationratio','debttoincomeratio',
            'missedpaymentcount','digitalactivityscore',
            'portfoliodiversityscore','gender_encoded',
            'occupation_encoded','pastinvestments_encoded'
        ]
        
        X = df[self.feature_columns].copy()
        X = X.fillna(X.median())
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
    
    # ------------------------------
    # Derived Features for New Users
    # ------------------------------
    def calculate_derived_features(self, user_input):
        seed_str = f"{user_input['age']}_{user_input['gender']}_{user_input['occupation']}_{user_input['creditscore']}_{user_input['pastinvestments']}"
        
        # 1. Credit Utilization Ratio
        if user_input['creditscore'] >= 750:
            creditutilizationratio = deterministic_random(seed_str+"1",0.05,0.20)
        elif user_input['creditscore'] >= 650:
            creditutilizationratio = deterministic_random(seed_str+"2",0.20,0.40)
        else:
            creditutilizationratio = deterministic_random(seed_str+"3",0.40,0.70)
        
        # 2. Debt to Income Ratio
        monthly_income = user_input['annualincome'] / 12
        monthly_surplus = monthly_income - user_input['avgmonthlyspend']
        estimated_emi = max(0, monthly_surplus * 0.25)
        debttoincomeratio = estimated_emi / monthly_income
        
        # 3. Transaction Volatility
        if user_input['occupation']=='Salaried' and user_input['creditscore']>=700:
            transactionvolatility = deterministic_random(seed_str+"4",0.08,0.15)
        elif user_input['occupation']=='Self-employed':
            transactionvolatility = deterministic_random(seed_str+"5",0.20,0.35)
        else:
            transactionvolatility = deterministic_random(seed_str+"6",0.15,0.25)
        
        # 4. Spending Stability Index
        if user_input['savingsrate']>=0.30:
            spendingstabilityindex = deterministic_random(seed_str+"7",0.70,0.85)
        elif user_input['savingsrate']>=0.15:
            spendingstabilityindex = deterministic_random(seed_str+"8",0.55,0.70)
        else:
            spendingstabilityindex = deterministic_random(seed_str+"9",0.40,0.55)
        
        # 5. Missed Payment Count
        if user_input['creditscore']>=750:
            missedpaymentcount = 0
        elif user_input['creditscore']>=650:
            missedpaymentcount = int(deterministic_random(seed_str+"10",0,1.99))
        else:
            missedpaymentcount = int(deterministic_random(seed_str+"11",1,3.99))
        
        # 6. Digital Activity Score
        if user_input['age']<30:
            digitalactivityscore = deterministic_random(seed_str+"12",70,85)
        elif user_input['age']<45:
            digitalactivityscore = deterministic_random(seed_str+"13",55,75)
        else:
            digitalactivityscore = deterministic_random(seed_str+"14",40,60)
        if user_input['citytier']==1:
            digitalactivityscore += 10
        if user_input['creditscore']<650:
            digitalactivityscore -=10
        digitalactivityscore = max(0,min(100,digitalactivityscore))
        
        # 7. Portfolio Diversity Score
        investment_types = user_input['pastinvestments']
        if investment_types=='None':
            portfoliodiversityscore = 0
        elif '_' in investment_types or ',' in investment_types:
            portfoliodiversityscore = deterministic_random(seed_str+"15",60,80)
        else:
            portfoliodiversityscore = deterministic_random(seed_str+"16",30,50)
        
        return {
            'transactionvolatility': transactionvolatility,
            'spendingstabilityindex': spendingstabilityindex,
            'creditutilizationratio': creditutilizationratio,
            'debttoincomeratio': debttoincomeratio,
            'missedpaymentcount': missedpaymentcount,
            'digitalactivityscore': digitalactivityscore,
            'portfoliodiversityscore': portfoliodiversityscore
        }
    
    # ------------------------------
    # Train Model
    # ------------------------------
    def train(self, df):
        df = self.create_risk_labels(df)
        X = self.preprocess_features(df, is_training=True)
        y = df['risk_label']
        
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Use custom SMOTE implementation
        smote = SimpleSMOTE(k_neighbors=3, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Sample weights
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (len(class_counts)*count) for i,count in enumerate(class_counts)}
        sample_weights = np.array([class_weights[label] for label in y_train_balanced])
        
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3
        )
        
        self.model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights, verbose=False)
        
        y_pred = self.model.predict(X_test)
        print("\n✅ Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))

        output_path = "customer_risk_output.csv"
        df.to_csv(output_path, index=False)
        print(f"📄 Saved risk-labeled dataset to: {output_path}")
        
        return self.model
    
    # ------------------------------
    # Predict New User
    # ------------------------------
    def predict_new_user(self, user_input):
        if self.model is None:
            raise ValueError("Model not trained! Load or train model first.")
        
        derived = self.calculate_derived_features(user_input)
        complete_data = {**user_input, **derived}
        pred_df = pd.DataFrame([complete_data])
        X = self.preprocess_features(pred_df, is_training=False)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        class_order = self.label_encoders['target'].classes_
        prob_dict = dict(zip(class_order, probabilities))
        risk_label = self.label_encoders['target'].inverse_transform([prediction])[0]
        
        print(f"\n🎯 Risk Level: {risk_label.upper()}")
        print("📊 Confidence Scores:")
        for cls in ['high','medium','low']:
            print(f"  {cls.capitalize()} Risk: {prob_dict.get(cls,0):.1%}")
        
        return {
            'risk_level': risk_label,
            'confidence_high': prob_dict.get('high',0),
            'confidence_medium': prob_dict.get('medium',0),
            'confidence_low': prob_dict.get('low',0)
        }
    
    # ------------------------------
    # Save / Load Model
    # ------------------------------
    def save_model(self, path='risk_model.pkl'):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        with open(path,'wb') as f:
            pickle.dump(model_data,f)
        print(f"\n💾 Model saved to {path}")
    
    def load_model(self, path='risk_model.pkl'):
        with open(path,'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"✅ Model loaded from {path}")

# ------------------------------
# USAGE DEMO
# ------------------------------
if __name__=="__main__":
    risk_model = RiskProfilingModel()
    
    # Load CSVs
    df = risk_model.load_data(
        r"D:\Reco\data\customerinfo.csv",
        r"D:\Reco\data\customerbehavior.csv"
    )
    
    # Train and save
    risk_model.train(df)
    risk_model.save_model('risk_model.pkl')
    
    print("\n" + "="*50)
    print("DEMO: PREDICT FOR NEW USER")
    print("="*50)
    
    new_user = {
        'age': 41,
        'gender': 'F',
        'citytier': 1,
        'annualincome': 900000,
        'occupation': 'Business',
        'creditscore': 580,
        'avgmonthlyspend': 70000,
        'savingsrate': 0.08,
        'investmentamountlastyear': 5000,
        'pastinvestments': 'insurance'
    }
    
    result = risk_model.predict_new_user(new_user)

