"""
Recommendation Routes
"""
from fastapi import APIRouter, HTTPException
from app.api.dependencies import stocks_data, funds_data, insurance_data, admin_logs
# Import get_user_profile function
from app.api.routes import users as users_route
from app.utils.helpers import cosine_similarity, add_log
from app.models.schemas import ExplanationRequest, IndividualExplanationRequest
from app.api.main import get_user_recommendations
import os
import aiohttp
import asyncio
from datetime import datetime
from app.core.config import GROQ_MODEL, GROQ_API_KEY, GROQ_API_URL
from typing import Dict, List

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int, top_k: int = 10):
    """Get stock, mutual fund, and insurance recommendations for a user"""
    
    if stocks_data is None or funds_data is None:
        raise HTTPException(status_code=404, detail="Stocks or Funds data not loaded")
    
    # Get user profile (this will raise 404 if user not found)
    try:
        user_profile = await users_route.get_user_profile(user_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        add_log(admin_logs, "GET_RECOMMENDATIONS", "ERROR", f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading user profile: {str(e)}")
    
    try:
        recommendations = get_user_recommendations(user_profile, top_k)
    except Exception as e:
        add_log(admin_logs, "GET_RECOMMENDATIONS", "ERROR", f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
    result = {
        "user_id": user_profile['user_id'],
        "user_metadata": user_profile['metadata'],
        "top_stock_recommendations": recommendations['stocks'],
        "top_mutual_fund_recommendations": recommendations['mutual_funds'],
        "top_insurance_recommendations": recommendations['insurance']
    }
    
    add_log(admin_logs, "GET_RECOMMENDATIONS", "SUCCESS", f"Generated recommendations for {user_id}")
    return result


@router.post("/explain")
async def explain_recommendations(request: ExplanationRequest):
    """Generate LLM explanation for recommendations"""
    
    try:
        add_log(admin_logs, "EXPLAIN", "PROCESSING", "Generating explanation for user")
        
        explanation = await generate_llm_explanation(
            user_profile=request.user_profile,
            top_stocks=request.top_stocks,
            top_mutual_funds=request.top_mutual_funds
        )
        
        add_log(admin_logs, "EXPLAIN", "SUCCESS", "Explanation generated successfully")
        
        return {
            "explanation": explanation,
            "status": "success",
            "metadata": {
                "model": GROQ_MODEL,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_risk": request.user_profile['risk_label']
            }
        }
    except Exception as e:
        add_log(admin_logs, "EXPLAIN", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-individual")
async def explain_individual_recommendation(request: IndividualExplanationRequest):
    """Generate LLM explanation for individual stock or mutual fund"""
    
    try:
        add_log(admin_logs, "EXPLAIN_INDIVIDUAL", "PROCESSING", 
                f"Generating explanation for {request.item_type}")
        
        explanation = await generate_individual_llm_explanation(
            user_profile=request.user_profile,
            item_type=request.item_type,
            item_data=request.item_data,
            admin_logs=admin_logs
        )
        
        add_log(admin_logs, "EXPLAIN_INDIVIDUAL", "SUCCESS", 
                f"Explanation generated for {request.item_type}")
        
        return {
            "explanation": explanation,
            "status": "success",
            "metadata": {
                "model": GROQ_MODEL,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_risk": request.user_profile['risk_label']
            }
        }
    except Exception as e:
        add_log(admin_logs, "EXPLAIN_INDIVIDUAL", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))


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

async def generate_llm_explanation(
    user_profile: Dict,
    top_stocks: List[Dict],
    top_mutual_funds: List[Dict]
) -> str:
    """Generate explanation using Groq API"""
    
    # Format stocks for prompt
    stocks_text = "\n".join([
        f"- {s['symbol']} ({s['company_name']}): Match Score {s['similarity_score']*100:.1f}%, "
        f"Sector: {s['metadata'].get('sector', 'N/A')}"
        for s in top_stocks[:5]
    ])
    
    # Format mutual funds for prompt
    funds_text = "\n".join([
        f"- {f['fund_name'][:60]}: Match Score {f['similarity_score']*100:.1f}%, "
        f"Category: {f['metadata'].get('category', 'N/A')}"
        for f in top_mutual_funds[:5]
    ])
    
    # Create the prompt
    prompt = f"""You are a financial insights assistant inside Raj's investment app.
Your task is to explain WHY the recommended stocks and mutual funds match the user's profile.

Guidelines:
- Use simple, human-friendly language.
- Explain in short bullet points (4-6 points maximum).
- Mention factors like risk level, volatility, goal alignment, sector stability, and past consistency.
- Never give financial advice or guaranteed returns.
- Keep the answer within 6-7 lines total.
- Add 1-2 generic caution notes about market risk at the end.

User Profile:
- Age: {user_profile['age']} years
- Occupation: {user_profile['occupation']}
- Risk Profile: {user_profile['risk_label'].upper()}
- Risk Score: {user_profile['risk_score']}
- Annual Income: ₹{user_profile['income']:,.0f}
- Credit Score: {user_profile['credit_score']}
- Savings Rate: {user_profile['savings_rate']*100:.1f}%
- Debt-to-Income Ratio: {user_profile['debt_to_income']*100:.1f}%
- Digital Activity Score: {user_profile['digital_activity']:.0f}/100
- Portfolio Diversity Score: {user_profile['portfolio_diversity']:.0f}/100

Top Stock Recommendations:
{stocks_text}

Top Mutual Fund Recommendations:
{funds_text}

Provide a clear, short explanation the user can understand. Focus on why these recommendations suit their profile."""

    try:
        if not GROQ_API_KEY:
            add_log(admin_logs, "LLM_EXPLAIN", "WARNING", "Groq API key not configured, using fallback")
            return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful financial insights assistant. Provide clear, concise explanations without giving specific financial advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9
            }
            
            async with session.post(
                GROQ_API_URL, 
                json=payload, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    explanation = result['choices'][0]['message']['content'].strip()
                    
                    if not explanation:
                        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
                    
                    add_log("LLM_EXPLAIN", "SUCCESS", f"Generated explanation using {GROQ_MODEL}")
                    return explanation
                else:
                    error_text = await response.text()
                    add_log("LLM_EXPLAIN", "ERROR", f"Groq API error {response.status}: {error_text}")
                    return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except asyncio.TimeoutError:
        add_log("LLM_EXPLAIN", "ERROR", "Groq API timeout")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except Exception as e:
        add_log("LLM_EXPLAIN", "ERROR", f"Groq error: {str(e)}")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)

async def generate_individual_llm_explanation(
    user_profile: Dict,
    item_type: str,
    item_data: Dict,
    admin_logs: List
) -> str:
    """Generate explanation for individual stock or mutual fund using Groq"""
    
    # Format item details based on type
    if item_type == "stock":
        item_name = f"{item_data.get('symbol', 'N/A')} ({item_data.get('company_name', 'N/A')})"
        item_details = f"""
Stock Details:
- Symbol: {item_data.get('symbol', 'N/A')}
- Company: {item_data.get('company_name', 'N/A')}
- Match Score: {item_data.get('similarity_score', 0)*100:.1f}%
- Sector: {item_data.get('metadata', {}).get('sector', 'N/A')}
- Market Cap: {item_data.get('metadata', {}).get('market_cap', 'N/A')}
"""
    else:  # mutual_fund
        item_name = item_data.get('fund_name', 'N/A')
        item_details = f"""
Mutual Fund Details:
- Fund Name: {item_data.get('fund_name', 'N/A')}
- Match Score: {item_data.get('similarity_score', 0)*100:.1f}%
- Category: {item_data.get('metadata', {}).get('category', 'N/A')}
- AUM: ₹{item_data.get('metadata', {}).get('aum', 'N/A')} Cr
"""
    
    prompt = f"""You are a financial insights assistant inside Raj's investment app.
Your task is to explain WHY this specific {item_type.replace('_', ' ')} is recommended for this user.

Guidelines:
- Use simple, human-friendly language.
- Explain in 3-4 short bullet points.
- Focus on why THIS specific investment matches the user's profile.
- Mention the match score and what makes it a good fit.
- Never give financial advice or guaranteed returns.
- Keep it brief and focused (4-5 lines total).
- Add 1 generic caution note about market risk at the end.

User Profile:
- Age: {user_profile['age']} years
- Occupation: {user_profile['occupation']}
- Risk Profile: {user_profile['risk_label'].upper()}
- Risk Score: {user_profile['risk_score']}
- Annual Income: ₹{user_profile['income']:,.0f}
- Credit Score: {user_profile['credit_score']}
- Savings Rate: {user_profile['savings_rate']*100:.1f}%
- Debt-to-Income Ratio: {user_profile['debt_to_income']*100:.1f}%
- Portfolio Diversity Score: {user_profile['portfolio_diversity']:.0f}/100

{item_details}

Explain specifically why {item_name} is recommended for this user. Focus on the match and alignment with their profile."""

    try:
        if not GROQ_API_KEY:
            add_log(admin_logs, "LLM_INDIVIDUAL", "WARNING", "Groq API key not configured, using fallback")
            return generate_fallback_individual_explanation(user_profile, item_type, item_data)
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful financial insights assistant. Provide clear, concise explanations without giving specific financial advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 300,
                "top_p": 0.9
            }
            
            async with session.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    explanation = result['choices'][0]['message']['content'].strip()
                    
                    if not explanation:
                        return generate_fallback_individual_explanation(user_profile, item_type, item_data)
                    
                    add_log(admin_logs, "LLM_INDIVIDUAL", "SUCCESS", f"Generated individual explanation using {GROQ_MODEL}")
                    return explanation
                else:
                    error_text = await response.text()
                    add_log(admin_logs, "LLM_INDIVIDUAL", "ERROR", f"Groq API error {response.status}: {error_text}")
                    return generate_fallback_individual_explanation(user_profile, item_type, item_data)
    
    except asyncio.TimeoutError:
        add_log(admin_logs, "LLM_INDIVIDUAL", "ERROR", "Groq API timeout")
        return generate_fallback_individual_explanation(user_profile, item_type, item_data)
    
    except Exception as e:
        add_log(admin_logs, "LLM_INDIVIDUAL", "ERROR", f"Groq error: {str(e)}")
        return generate_fallback_individual_explanation(user_profile, item_type, item_data)

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

