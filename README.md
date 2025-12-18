# Financial Recommendation System

Production-ready financial recommendation system with FastAPI backend and Streamlit frontend.

## Project Structure

```
Feildingsetpoc/
├── app/                      # Main application package
│   ├── api/                  # FastAPI routes and endpoints
│   │   └── main.py          # Main FastAPI application
│   ├── core/                 # Core configuration
│   │   ├── config.py        # Application configuration
│   │   └── constants.py     # Application constants
│   ├── models/              # Pydantic models
│   │   └── schemas.py       # API request/response schemas
│   ├── services/            # Business logic services
│   │   └── risk_service.py  # Risk assessment service
│   └── utils/               # Utility functions
│       ├── helpers.py       # Helper functions
│       └── feature_engineering.py  # Feature engineering
├── data/                     # Data files (CSV, JSON)
├── models/                    # ML model files (.pkl)
├── scripts/                  # Scripts
│   ├── data_injection/      # Data gathering and processing scripts
│   └── training/            # Model training scripts
├── logs/                     # Log files
├── main.py                   # FastAPI entry point
├── app.py                    # Streamlit entry point
└── requirements.txt         # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

3. Ensure data files are in the `data/` directory:
   - `test_users_BANK.csv`
   - `engineered_stocks.json`
   - `engineered_funds.json`
   - `engineered_insurance.json`
   - `health_policies.json`

4. Ensure model files are in the `models/` directory:
   - `risk_model.pkl`
   - `xgb_model_synthetic.pkl`
   - `xgb_scaler_synthetic.pkl`
   - `xgb_label_encoder_synthetic.pkl`
   - `xgb_feature_names_synthetic.pkl`

## Running the Application

### FastAPI Backend
```bash
python main.py
# or
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit Frontend
```bash
streamlit run app.py
```

## API Endpoints

- `GET /` - API information
- `GET /api/users` - Get all users
- `GET /api/user/{user_id}` - Get user profile
- `GET /api/recommendations/{user_id}` - Get recommendations
- `GET /api/insurance/predict/{user_id}` - Insurance predictions
- `POST /api/explain` - Generate LLM explanations
- `GET /api/admin/logs` - Admin logs
- `GET /api/scheduler/status` - Scheduler status

## Features

- **Risk Profiling**: ML-based risk assessment
- **Recommendations**: Stock, mutual fund, and insurance recommendations
- **Insurance Predictions**: XGBoost-powered health insurance predictions
- **AI Explanations**: Groq LLM-powered explanations
- **Automated Scheduler**: Scheduled data refresh

## Notes

- All logic has been preserved from the original implementation
- File paths have been updated to use the new structure
- Data files should be placed in the `data/` directory
- Model files should be placed in the `models/` directory

