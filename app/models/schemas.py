"""
Pydantic Models and Schemas
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class UserProfile(BaseModel):
    customer_id: str
    age: int
    gender: str
    citytier: int
    annualincome: float
    occupation: str
    creditscore: int
    avgmonthlyspend: float
    savingsrate: float
    investmentamountlastyear: float
    pastinvestments: str


class UserResponse(BaseModel):
    user_id: str
    engineered_vector: List[float]
    metadata: Dict
    derived_features: Dict
    notes: str


class RecommendationResponse(BaseModel):
    user_id: str
    user_metadata: Dict
    top_stock_recommendations: List[Dict]
    top_mutual_fund_recommendations: List[Dict]


class AdminLog(BaseModel):
    timestamp: str
    action: str
    status: str
    details: str


class ExplanationRequest(BaseModel):
    user_profile: Dict
    top_stocks: List[Dict]
    top_mutual_funds: List[Dict]


class IndividualExplanationRequest(BaseModel):
    user_profile: Dict
    item_type: str  # "stock" or "mutual_fund"
    item_data: Dict


class ExplanationResponse(BaseModel):
    explanation: str
    status: str
    metadata: Dict


class InsurancePredictionResponse(BaseModel):
    customer_id: str
    user_metadata: Dict
    predicted_policies: List[Dict]
    top_policy: Dict

