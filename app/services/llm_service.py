"""
LLM Service - Groq API Integration
"""
import aiohttp
import asyncio
from typing import Dict, List
from app.core.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL
from app.api.main import admin_logs, add_log
from app.api.main import generate_fallback_explanation, generate_fallback_individual_explanation


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
                    
                    add_log(admin_logs, "LLM_EXPLAIN", "SUCCESS", f"Generated explanation using {GROQ_MODEL}")
                    return explanation
                else:
                    error_text = await response.text()
                    add_log(admin_logs, "LLM_EXPLAIN", "ERROR", f"Groq API error {response.status}: {error_text}")
                    return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except asyncio.TimeoutError:
        add_log(admin_logs, "LLM_EXPLAIN", "ERROR", "Groq API timeout")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)
    
    except Exception as e:
        add_log(admin_logs, "LLM_EXPLAIN", "ERROR", f"Groq error: {str(e)}")
        return generate_fallback_explanation(user_profile, top_stocks, top_mutual_funds)


async def generate_individual_llm_explanation(
    user_profile: Dict,
    item_type: str,
    item_data: Dict
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

