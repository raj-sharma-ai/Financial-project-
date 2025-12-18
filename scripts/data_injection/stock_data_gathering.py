#stock_data_gathering.py
from scipy import stats
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
import os
import requests
from io import StringIO

warnings.filterwarnings('ignore')





SECTOR_DATABASE = {
    # IT SECTOR (50+)
    'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
    'LTIM': 'IT', 'PERSISTENT': 'IT', 'COFORGE': 'IT', 'LTTS': 'IT', 'MPHASIS': 'IT',
    'SONATSOFTW': 'IT', 'TATAELXSI': 'IT', 'CYIENT': 'IT', 'KPITTECH': 'IT',
    'ZENSARTECH': 'IT', 'TATATECH': 'IT', 'BSOFT': 'IT', 'HAPPSTMNDS': 'IT',
    'INTELLECT': 'IT', 'UNITDSPR': 'IT', 'SAGILITY': 'IT', 'CREDITACC': 'IT',
    'TRITURBINE': 'IT', 'TECHNOE': 'IT', 'GRAVITA': 'IT', 'GRAPHITE': 'IT',
    'GODIGIT': 'IT', 'TITAGARH': 'IT', 'RITES': 'IT', 'ITI': 'IT',
    'ECLERX': 'IT', 'ABCAPITAL': 'IT', 'ITCHOTELS': 'IT',
    
    # BANKING (35+)
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
    'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking', 'FEDERALBNK': 'Banking', 'BANDHANBNK': 'Banking',
    'IDFCFIRSTB': 'Banking', 'PNB': 'Banking', 'BANKBARODA': 'Banking', 'CANBK': 'Banking',
    'UNIONBANK': 'Banking', 'YESBANK': 'Banking', 'RBLBANK': 'Banking', 'AUBANK': 'Banking',
    'BANKINDIA': 'Banking', 'J&KBANK': 'Banking', 'MAHABANK': 'Banking', 'UCOBANK': 'Banking',
    'CENTRALBK': 'Banking', 'KARURVYSYA': 'Banking', 'IDBI': 'Banking',
    
    # FINANCIAL SERVICES (45+)
    'BAJFINANCE': 'Financial Services', 'BAJAJFINSV': 'Financial Services', 
    'HDFCLIFE': 'Financial Services', 'ICICIPRULI': 'Financial Services',
    'SBILIFE': 'Financial Services', 'LICI': 'Financial Services', 'CHOLAFIN': 'Financial Services',
    'SHRIRAMFIN': 'Financial Services', 'SBICARD': 'Financial Services', 'BAJAJHFL': 'Financial Services',
    'JIOFIN': 'Financial Services', 'CDSL': 'Financial Services', 'CAMS': 'Financial Services',
    'MUTHOOTFIN': 'Financial Services', 'LICHSGFIN': 'Financial Services', 'M&MFIN': 'Financial Services',
    'KFINTECH': 'Financial Services', 'ZYDUSLIFE': 'Financial Services', 'SAILIFE': 'Financial Services',
    'SUNDARMFIN': 'Financial Services', 'FINPIPE': 'Financial Services', 'JMFINANCIL': 'Financial Services',
    'SKFINDIA': 'Financial Services', 'SKFINDUS': 'Financial Services', 'FINCABLES': 'Financial Services',
    'CANFINHOME': 'Financial Services', 'MANAPPURAM': 'Financial Services', 'AADHARHFC': 'Financial Services',
    'POONAWALLA': 'Financial Services', 'IIFL': 'Financial Services', 'APTUS': 'Financial Services',
    '360ONE': 'Financial Services', 'ICICIPRULI': 'Financial Services', 'SBFC': 'Financial Services',
    'MOTILALOFS': 'Financial Services', 'HOMEFIRST': 'Financial Services', 'AAVAS': 'Financial Services',
    
    # ENERGY & POWER (40+)
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'IOC': 'Energy',
    'NTPC': 'Energy', 'POWERGRID': 'Energy', 'ADANIGREEN': 'Energy', 'TATAPOWER': 'Energy',
    'COALINDIA': 'Energy', 'ADANIPOWER': 'Energy', 'ADANIENSOL': 'Energy', 'JSWENERGY': 'Energy',
    'OIL': 'Energy', 'TORNTPOWER': 'Energy', 'RPOWER': 'Energy', 'POWERINDIA': 'Energy',
    'GUJGASLTD': 'Energy', 'JPPOWER': 'Energy', 'IGL': 'Energy', 'ATGL': 'Energy',
    'MGL': 'Energy', 'HINDPETRO': 'Energy', 'GAIL': 'Energy', 'NHPC': 'Energy',
    'SJVN': 'Energy', 'CESC': 'Energy', 'NTPCGREEN': 'Energy', 'SUZLON': 'Energy',
    'PREMIERENE': 'Energy', 'INOXWIND': 'Energy', 'ACMESOLAR': 'Energy', 'NLCINDIA': 'Energy',
    'PETRONET': 'Energy', 'GSPL': 'Energy', 'CHENNPETRO': 'Energy', 'MRPL': 'Energy',
    'RCF': 'Energy', 'FACT': 'Energy', 'WAAREEENER': 'Energy',
    
    # FMCG (25+)
    'ITC': 'FMCG', 'HINDUNILVR': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
    'DABUR': 'FMCG', 'MARICO': 'FMCG', 'TATACONSUM': 'FMCG', 'GODREJCP': 'FMCG',
    'VBL': 'FMCG', 'UBL': 'FMCG', 'AWL': 'FMCG', 'COLPAL': 'FMCG', 'GILLETTE': 'FMCG',
    'RADICO': 'FMCG', 'PGHH': 'FMCG', 'EMAMILTD': 'FMCG', 'JYOTHYLAB': 'FMCG',
    'GODREJAGRO': 'FMCG', 'BIKAJI': 'FMCG', 'LTFOODS': 'FMCG', 'BALRAMCHIN': 'FMCG',
    'PATANJALI': 'FMCG', 'JUBLFOOD': 'FMCG', 'DEVYANI': 'FMCG',
    
    # AUTOMOBILE (30+)
    'MARUTI': 'Automobile', 'TATAMOTORS': 'Automobile', 'M&M': 'Automobile',
    'BAJAJ-AUTO': 'Automobile', 'HEROMOTOCO': 'Automobile', 'EICHERMOT': 'Automobile',
    'TVSMOTOR': 'Automobile', 'MOTHERSON': 'Automobile', 'MRF': 'Automobile',
    'APOLLOTYRE': 'Automobile', 'BOSCHLTD': 'Automobile', 'EXIDEIND': 'Automobile',
    'HYUNDAI': 'Automobile', 'CEATLTD': 'Automobile', 'BALKRISIND': 'Automobile',
    'JKTYRE': 'Automobile', 'ASHOKLEY': 'Automobile', 'ESCORTS': 'Automobile',
    'MSUMI': 'Automobile', 'FORCEMOT': 'Automobile', 'ENDURANCE': 'Automobile',
    'MAHSEAMLES': 'Automobile', 'MAHSCOOTER': 'Automobile', 'ASAHIINDIA': 'Automobile',
    'ZFCVINDIA': 'Automobile', 'UNOMINDA': 'Automobile', 'OLECTRA': 'Automobile',
    'ATHERENERG': 'Automobile', 'SONACOMS': 'Automobile',
    
    # PHARMACEUTICALS & HEALTHCARE (40+)
    'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'DIVISLAB': 'Pharma', 'CIPLA': 'Pharma',
    'AUROPHARMA': 'Pharma', 'BIOCON': 'Pharma', 'TORNTPHARM': 'Pharma', 'LUPIN': 'Pharma',
    'APOLLOHOSP': 'Pharma', 'MAXHEALTH': 'Pharma', 'FORTIS': 'Pharma', 'ALKEM': 'Pharma',
    'WOCKPHARMA': 'Pharma', 'PPLPHARMA': 'Pharma', 'JUBLPHARMA': 'Pharma', 'MEDANTA': 'Pharma',
    'POLYMED': 'Pharma', 'NATCOPHARM': 'Pharma', 'LAURUSLABS': 'Pharma', 'GRANULES': 'Pharma',
    'GLENMARK': 'Pharma', 'NEULANDLAB': 'Pharma', 'GLAND': 'Pharma', 'JBCHEPHARM': 'Pharma',
    'ERIS': 'Pharma', 'EMCURE': 'Pharma', 'SYNGENE': 'Pharma', 'CAPLIPOINT': 'Pharma',
    'PFIZER': 'Pharma', 'GLAXO': 'Pharma', 'SANOFI': 'Pharma', 'ABBOTINDIA': 'Pharma',
    'ASTRAZEN': 'Pharma', 'IPCALAB': 'Pharma', 'AJANTPHARM': 'Pharma', 'CONCORDBIO': 'Pharma',
    'AKUMS': 'Pharma', 'LALPATHLAB': 'Pharma', 'METROPOLIS': 'Pharma', 'ASTERDM': 'Pharma',
    'RAINBOW': 'Pharma', 'KIMS': 'Pharma', 'AGARWALEYE': 'Pharma',
    
    # METALS & MINING (25+)
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals',
    'JINDALSTEL': 'Metals', 'SAIL': 'Metals', 'NMDC': 'Metals', 'HINDZINC': 'Metals',
    'HINDCOPPER': 'Metals', 'NATIONALUM': 'Metals', 'WELCORP': 'Metals', 'JINDALSAW': 'Metals',
    'SHYAMMETL': 'Metals', 'JSL': 'Metals', 'APARINDS': 'Metals', 'RKFORGE': 'Metals',
    'HEG': 'Metals', 'CARBORUNIV': 'Metals', 'GMDCLTD': 'Metals', 'MOIL': 'Metals',
    
    # CEMENT (15+)
    'ULTRACEMCO': 'Cement', 'GRASIM': 'Cement', 'AMBUJACEM': 'Cement', 'ACC': 'Cement',
    'SHREECEM': 'Cement', 'DALMIACEM': 'Cement', 'JKCEMENT': 'Cement', 'RAMCOCEM': 'Cement',
    'DALBHARAT': 'Cement', 'NUVOCO': 'Cement', 'HEIDELBERG': 'Cement', 'INDIACEM': 'Cement',
    'BIRLAMONEY': 'Cement', 'JKLAKSHMI': 'Cement',
    
    # INFRASTRUCTURE & CONSTRUCTION (40+)
    'LT': 'Infrastructure', 'ADANIPORTS': 'Infrastructure', 'DLF': 'Infrastructure',
    'GODREJPROP': 'Infrastructure', 'OBEROIRLTY': 'Infrastructure', 'PRESTIGE': 'Infrastructure',
    'IRFC': 'Infrastructure', 'RVNL': 'Infrastructure', 'IRCTC': 'Infrastructure',
    'NBCC': 'Infrastructure', 'CONCOR': 'Infrastructure', 'CGPOWER': 'Infrastructure',
    'ENRIN': 'Infrastructure', 'ABB': 'Infrastructure', 'INDHOTEL': 'Infrastructure',
    'DBREALTY': 'Infrastructure', 'BRIGADE': 'Infrastructure', 'SOBHA': 'Infrastructure',
    'LODHA': 'Infrastructure', 'PHOENIXLTD': 'Infrastructure', 'HUDCO': 'Infrastructure',
    'COCHINSHIP': 'Infrastructure', 'IRCON': 'Infrastructure', 'NCC': 'Infrastructure',
    'AFCONS': 'Infrastructure', 'BEML': 'Infrastructure', 'JSWINFRA': 'Infrastructure',
    'RELINFRA': 'Infrastructure', 'TMPV': 'Infrastructure', 'ETERNAL': 'Infrastructure',
    'KEC': 'Infrastructure', 'IRB': 'Infrastructure', 'ANANTRAJ': 'Infrastructure',
    'SCI': 'Infrastructure', 'ACE': 'Infrastructure', 'GMRAIRPORT': 'Infrastructure',
    
    # CHEMICALS (25+)
    'UPL': 'Chemicals', 'PIDILITIND': 'Chemicals', 'SRF': 'Chemicals',
    'DEEPAKNTR': 'Chemicals', 'TATACHEM': 'Chemicals', 'ATUL': 'Chemicals',
    'ASIANPAINT': 'Chemicals', 'AARTIIND': 'Chemicals', 'DEEPAKFERT': 'Chemicals',
    'CHAMBLFERT': 'Chemicals', 'SUMICHEM': 'Chemicals', 'FLUOROCHEM': 'Chemicals',
    'ALKYLAMINE': 'Chemicals', 'NAVINFLUOR': 'Chemicals', 'ASTRAL': 'Chemicals',
    'CLEAN': 'Chemicals', 'COROMANDEL': 'Chemicals', 'GNFC': 'Chemicals',
    'BASF': 'Chemicals', 'AKZOINDIA': 'Chemicals', 'BAYERCROP': 'Chemicals',
    'BERGEPAINT': 'Chemicals', 'EIDPARRY': 'Chemicals', 'DCMSHRIRAM': 'Chemicals',
    
    # DEFENCE (10+)
    'HAL': 'Defence', 'BEL': 'Defence', 'BDL': 'Defence', 'GRSE': 'Defence',
    'COCHINSHIP': 'Defence', 'MAZAGON': 'Defence', 'MAZDOCK': 'Defence',
    'DATAPATTNS': 'Defence', 'SOLARINDS': 'Defence', 'GESHIP': 'Defence',
    
    # RETAIL & ECOMMERCE (15+)
    'TRENT': 'Retail', 'ABFRL': 'Retail', 'ZOMATO': 'Retail', 'NYKAA': 'Retail',
    'PAYTM': 'Retail', 'DELHIVERY': 'Retail', 'DMART': 'Retail', 'SWIGGY': 'Retail',
    'FIRSTCRY': 'Retail', 'PAGEIND': 'Retail', 'BATAINDIA': 'Retail', 'MANYAVAR': 'Retail',
    'CAMPUS': 'Retail', 'RELAXO': 'Retail',
    
    # CONSUMER DURABLES (20+)
    'HAVELLS': 'Consumer Durables', 'DIXON': 'Consumer Durables', 'TITAN': 'Consumer Durables',
    'VOLTAS': 'Consumer Durables', 'WHIRLPOOL': 'Consumer Durables', 'CROMPTON': 'Consumer Durables',
    'POLYCAB': 'Consumer Durables', 'BLUESTARCO': 'Consumer Durables', 'VGUARD': 'Consumer Durables',
    'SYMPHONY': 'Consumer Durables', 'AMBER': 'Consumer Durables', 'KAJARIACER': 'Consumer Durables',
    'CERA': 'Consumer Durables', 'CENTURYPLY': 'Consumer Durables', 'DOMS': 'Consumer Durables',
    'SAFARI': 'Consumer Durables', 'VIPIND': 'Consumer Durables',
    
    # TELECOM & MEDIA (15+)
    'BHARTIARTL': 'Telecom', 'IDEA': 'Telecom', 'BHARTIHEXA': 'Telecom',
    'TTML': 'Telecom', 'HFCL': 'Telecom', 'RAILTEL': 'Telecom',
    'ZEE': 'Media', 'SUNTV': 'Media', 'PVRINOX': 'Media', 'NETWORK18': 'Media',
    'TV18BRDCST': 'Media', 'ZEEL': 'Media', 'SAREGAMA': 'Media',
    
    # AVIATION
    'INDIGO': 'Aviation',
    
    # TEXTILES (10+)
    'VTL': 'Textiles', 'RAYMOND': 'Textiles', 'TRIDENT': 'Textiles',
    'WELSPUNLIV': 'Textiles', 'ALOKINDS': 'Textiles', 'KPRMILL': 'Textiles',
    
    # LOGISTICS (10+)
    'BLUEDART': 'Logistics', 'TCI': 'Logistics', 'MAHLOG': 'Logistics',
    'AEGISLOG': 'Logistics', 'TBOTEK': 'Logistics', 'VRL': 'Logistics',
    
    # HOSPITALITY (10+)
    'INDHOTEL': 'Hospitality', 'LEMONTREE': 'Hospitality', 'CHALET': 'Hospitality',
    'EIHOTEL': 'Hospitality', 'THELEELA': 'Hospitality',
    
    # OTHERS - Remaining stocks
    'RECLTD': 'Others', 'PFC': 'Others', 'ICICIGI': 'Others', 'NAUKRI': 'Others',
    'SIEMENS': 'Others', 'BHEL': 'Others', 'POLICYBZR': 'Others', 'INDUSTOWER': 'Others',
    'PIIND': 'Others', 'BHARATFORG': 'Others', 'OFSS': 'Others', 'APLAPOLLO': 'Others',
    'MANKIND': 'Others', 'TIINDIA': 'Others', 'SUPREMEIND': 'Others', 'MFSL': 'Others',
    'BSE': 'Others', 'LTF': 'Others', 'INDIANB': 'Others', 'IREDA': 'Others',
    'STARHEALTH': 'Others', 'JWL': 'Others', 'IFCI': 'Others', 'REDINGTON': 'Others',
    'IGIL': 'Others', 'SIGNATURE': 'Others', 'MCX': 'Others', 'CHOLAHLDNG': 'Others',
    'PCBL': 'Others', 'PNBHOUSING': 'Others', 'ARE&M': 'Others', 'CGCL': 'Others',
    'TEJASNET': 'Others', 'AFFLE': 'Others', 'KPIL': 'Others', 'FSL': 'Others',
    'BLS': 'Others', 'OLAELEC': 'Others', 'HBLENGINE': 'Others', 'IEX': 'Others',
    'JBMA': 'Others', 'FIVESTAR': 'Others', 'SWANCORP': 'Others', 'CASTROLIND': 'Others',
    'ANGELONE': 'Others', 'NUVAMA': 'Others', 'IKS': 'Others', 'ANANDRATHI': 'Others',
    'HSCL': 'Others', 'NH': 'Others', 'AEGISVOPAK': 'Others', 'ZENTEC': 'Others',
    'ABREL': 'Others', 'GVT&D': 'Others', 'CCL': 'Others', 'BBTC': 'Others',
    'KIRLOSBROS': 'Others', 'KIRLOSENG': 'Others', 'SYRMA': 'Others', 'RRKABEL': 'Others',
    'SAMMAANCAP': 'Others', 'AIAENG': 'Others', 'NAM-INDIA': 'Others', 'TBOTEK': 'Others',
    'TARIL': 'Others', 'SCHAEFFLER': 'Others', 'TIMKEN': 'Others', 'INDGN': 'Others',
    'ELGIEQUIP': 'Others', 'NETWEB': 'Others', 'JUBLINGREA': 'Others', 'USHAMART': 'Others',
    'HONAUT': 'Others', 'PRAJIND': 'Others', 'UTIAMC': 'Others', 'CUB': 'Others',
    'KSB': 'Others', 'APLLTD': 'Others', 'ABSLAMC': 'Others', 'TATAINVEST': 'Others',
    'ELECON': 'Others', 'MAPMYINDIA': 'Others', 'GPIL': 'Others', 'LINDEINDIA': 'Others',
    'GODREJIND': 'Others', 'HEXT': 'Others', 'COHANCE': 'Others', 'SUNDRMFAST': 'Others',
    'INOXINDIA': 'Others', 'VENTIVE': 'Others', 'VIJAYA': 'Others', 'THERMAX': 'Others',
    'RHIM': 'Others', 'WELSPUNLIV': 'Others', 'CRAFTSMAN': 'Others', 'SCHNEIDER': 'Others',
    'PTCIL': 'Others', 'GICRE': 'Others', 'AIIL': 'Others', 'LLOYDSME': 'Others',
    'NIVABUPA': 'Others', 'LATENTVIEW': 'Others', 'BLUEJET': 'Others', 'CHOICEIN': 'Others',
    'HONASA': 'Others', '3MINDIA': 'Others', 'SAPPHIRE': 'Others', 'ONESOURCE': 'Others',
    'MMTC': 'Others', 'MINDACORP': 'Others', 'NIACL': 'Others', 'NSLNISP': 'Others',
    'SARDAEN': 'Others', 'INDIAMART': 'Others', 'KAYNES': 'Others', 'PGEL': 'Others',
    'JYOTICNC': 'Others', 'TRIVENI': 'Others', 'NAVA': 'Others', 'ENGINERSIN': 'Others',
    'ADANIENT': 'Others', 'GAIL': 'Others', 'GODFRYPHLP': 'Others', 'VMPL': 'Others',
    'KALYANKJIL': 'Others', 'PATANJALI': 'Others', 'VMM': 'Others', 'INDIAMART': 'Others',
}





def get_sector_from_symbol(symbol):
    """
    Get sector from symbol with comprehensive keyword-based fallback.
    ALL SECTORS ARE MANDATORY - NO 'Others' category!
    """
    symbol = symbol.upper().strip()
    
    # Direct lookup first
    if symbol in SECTOR_DATABASE:
        return SECTOR_DATABASE[symbol]
    
    # ============ KEYWORD-BASED FALLBACK (ALL SECTORS) ============
    
    # BANKING
    if 'BANK' in symbol:
        return 'Banking'
    
    # FINANCIAL SERVICES
    if any(x in symbol for x in ['FIN', 'INSUR', 'INSURANCE', 'LIFE', 'CAPITAL', 
                                   'CARD', 'LEASING', 'LOAN', 'CREDIT', 'HOUSING']):
        return 'Financial Services'
    
    # IT
    if any(x in symbol for x in ['IT', 'TECH', 'SOFT', 'INFO', 'DATA', 'CYBER',
                                   'DIGITAL', 'CLOUD', 'INFY', 'SYSTEM']):
        return 'IT'
    
    # ENERGY & POWER
    if any(x in symbol for x in ['POWER', 'ENERGY', 'COAL', 'GAS', 'OIL', 'PETRO',
                                   'SOLAR', 'WIND', 'HYDRO', 'ELECTRIC', 'FUEL']):
        return 'Energy'
    
    # PHARMA & HEALTHCARE
    if any(x in symbol for x in ['PHARMA', 'DRUG', 'MED', 'HOSP', 'HEALTH', 'BIO',
                                   'LAB', 'CLINIC', 'CARE', 'LIFE']):
        return 'Pharma'
    
    # AUTOMOBILE
    if any(x in symbol for x in ['AUTO', 'MOTOR', 'TYRE', 'VEHICLE', 'CAR', 'BIKE',
                                   'WHEEL', 'FORGE', 'EICHER']):
        return 'Automobile'
    
    # METALS & MINING
    if any(x in symbol for x in ['STEEL', 'METAL', 'ZINC', 'COPPER', 'ALUMIN', 
                                   'IRON', 'MINING', 'ORE', 'ALLOY']):
        return 'Metals'
    
    # CEMENT
    if any(x in symbol for x in ['CEMENT', 'CONCRETE', 'ULTRATECH']):
        return 'Cement'
    
    # CHEMICALS
    if any(x in symbol for x in ['CHEM', 'FERT', 'FERTILIZER', 'PAINT', 'ALKALI',
                                   'ACID', 'POLYMER']):
        return 'Chemicals'
    
    # INFRASTRUCTURE & CONSTRUCTION
    if any(x in symbol for x in ['INFRA', 'CONST', 'BUILD', 'REAL', 'PROP', 'REALTY',
                                   'HOUSING', 'PORT', 'ROAD', 'BRIDGE', 'RAILWAY']):
        return 'Infrastructure'
    
    # FMCG
    if any(x in symbol for x in ['CONSUMER', 'FMCG', 'FOOD', 'BEVER', 'TOBACCO',
                                   'DAIRY', 'AGRO', 'MILLS']):
        return 'FMCG'
    
    # DEFENCE
    if any(x in symbol for x in ['DEFENCE', 'DEFENSE', 'AEROSPACE', 'WEAPON', 
                                   'ORDNANCE', 'MILITARY', 'HAL', 'BEL', 'BDL']):
        return 'Defence'
    
    # RETAIL & ECOMMERCE
    if any(x in symbol for x in ['RETAIL', 'MART', 'STORE', 'SHOP', 'ECOM', 
                                   'ECOMMERCE', 'ONLINE', 'ZOMATO', 'SWIGGY']):
        return 'Retail'
    
    # CONSUMER DURABLES
    if any(x in symbol for x in ['DURABLE', 'APPLIANCE', 'ELECTRONICS', 'FAN',
                                   'WIRE', 'CABLE', 'JEWEL', 'TITAN']):
        return 'Consumer Durables'
    
    # TELECOM & MEDIA
    if any(x in symbol for x in ['TELECOM', 'BHARTI', 'AIRTEL', 'IDEA', 'VODAFONE']):
        return 'Telecom'
    
    if any(x in symbol for x in ['MEDIA', 'BROADCAST', 'ENTERTAINMENT', 'ZEE', 
                                   'TV', 'FILM', 'MUSIC']):
        return 'Media'
    
    # TEXTILES
    if any(x in symbol for x in ['TEXTILE', 'FABRIC', 'COTTON', 'YARN', 'SPINNING',
                                   'GARMENT', 'APPAREL']):
        return 'Textiles'
    
    # LOGISTICS
    if any(x in symbol for x in ['LOGISTIC', 'CARGO', 'FREIGHT', 'TRANSPORT',
                                   'SHIPPING', 'DELIVERY']):
        return 'Logistics'
    
    # AVIATION
    if any(x in symbol for x in ['AVIATION', 'AIRLINE', 'AIRPORT', 'FLIGHT', 'INDIGO']):
        return 'Aviation'
    
    # HOSPITALITY
    if any(x in symbol for x in ['HOTEL', 'RESORT', 'HOSPITALITY', 'TOURISM', 'LEMON']):
        return 'Hospitality'
    
    # If nothing matches, return Others
    return 'Others'

# ============================================================================
# 🌐 STOCK LIST FETCHER
# ============================================================================

def fetch_all_nse_stocks():
    """Fetch comprehensive NSE stock list"""
    print("🚀 Fetching comprehensive NSE stock list...")
    
    all_stocks = {}
    
    indices = [
        "NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 50",
        "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100", "NIFTY 500",
        "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO",
        "NIFTY FMCG", "NIFTY METAL", "NIFTY ENERGY"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    for index in indices:
        try:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={index.replace(' ', '%20')}"
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            time.sleep(1)
            
            response = session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    symbol = item.get('symbol', '').strip()
                    if symbol and symbol not in all_stocks:
                        all_stocks[symbol] = {
                            'symbol': symbol,
                            'company_name': item.get('companyName', symbol)
                        }
                print(f"  ✅ {index}: {len(data.get('data', []))} stocks")
            else:
                print(f"  ⚠️  {index}: Failed")
        except Exception as e:
            print(f"  ⚠️  {index}: Error")
            continue
        
        time.sleep(0.5)
    
    # Add predefined stocks
    predefined = get_comprehensive_stock_list()
    for stock in predefined:
        symbol = stock['symbol']
        if symbol not in all_stocks:
            all_stocks[symbol] = stock
    
    stock_list = list(all_stocks.values())
    print(f"\n✅ Total unique stocks: {len(stock_list)}")
    return stock_list


def get_comprehensive_stock_list():
    """Predefined stock list"""
    stocks = []
    for symbol in SECTOR_DATABASE.keys():
        if symbol not in ['NIFTY 50', 'NIFTY NEXT 50', 'BANKNIFTY']:
            stocks.append({
                'symbol': symbol,
                'company_name': symbol
            })
    return stocks


def classify_market_cap(market_cap):
    """Classify by market cap"""
    if not market_cap or market_cap == 0:
        return 'Unknown'
    
    cap_in_crores = market_cap / 10_000_000
    
    if cap_in_crores >= 100_000:
        return 'Large Cap'
    elif cap_in_crores >= 25_000:
        return 'Mid Cap'
    else:
        return 'Small Cap'


# ============================================================================
# 📊 LIVE DATA ANALYZER
# ============================================================================

class LiveStockAnalyzer:
    
    def __init__(self, ticker, sector='Others'):
        self.ticker = ticker
        self.sector = sector
        self.stock = yf.Ticker(ticker)
        self.info = None
        self.hist = None
        
    def fetch_live_data(self):
        """Fetch live stock data"""
        try:
            self.info = self.stock.info
            self.hist = self.stock.history(period='5y')
            return len(self.hist) > 0
        except Exception:
            return False
    
    def get_basic_info(self):
        """Get basic info"""
        return {
            'symbol': self.info.get('symbol', ''),
            'company_name': self.info.get('longName', ''),
            'sector': self.sector,
            'market_cap': self.info.get('marketCap', 0),
            'market_cap_class': classify_market_cap(self.info.get('marketCap', 0)),
            'current_price': self.info.get('currentPrice', 0),
            'beta': round(self.info.get('beta', 1.0), 3) if self.info.get('beta') else 1.0,
        }
    
    def calculate_volatility(self):
        """Calculate volatility"""
        if self.hist is None or len(self.hist) < 90:
            return {'volatility_30d': None, 'volatility_90d': None}
        
        daily_returns = self.hist['Close'].pct_change().dropna()
        
        vol_30d = None
        if len(daily_returns) >= 30:
            vol_30d = round(daily_returns.tail(30).std() * np.sqrt(252) * 100, 2)
        
        vol_90d = None
        if len(daily_returns) >= 90:
            vol_90d = round(daily_returns.tail(90).std() * np.sqrt(252) * 100, 2)
        
        return {'volatility_30d': vol_30d, 'volatility_90d': vol_90d}
    
    def get_ohlc_data(self):
        """Get OHLC data"""
        if self.hist is None or len(self.hist) == 0:
            return {'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}
        
        latest = self.hist.iloc[-1]
        return {
            'open': round(latest['Open'], 2),
            'high': round(latest['High'], 2),
            'low': round(latest['Low'], 2),
            'close': round(latest['Close'], 2),
            'volume': int(latest['Volume'])
        }
    
    def calculate_returns(self):
        """Calculate returns"""
        if self.hist is None or len(self.hist) < 21:
            return {'return_1m': None, 'return_3m': None, 'return_6m': None, 
                    'return_1y': None, 'return_3y': None}
        
        current_price = self.hist['Close'].iloc[-1]
        returns = {}
        
        periods = {
            'return_1m': 21,
            'return_3m': 63,
            'return_6m': 126,
            'return_1y': 252,
            'return_3y': 630
        }
        
        for key, days in periods.items():
            max_lookback = int(len(self.hist) * 0.9)
            actual_days = min(days, max_lookback)
            
            if len(self.hist) >= actual_days and actual_days >= 21:
                past_price = self.hist['Close'].iloc[-actual_days]
                ret = ((current_price / past_price) - 1) * 100
                returns[key] = round(ret, 2)
            else:
                returns[key] = None
        
        return returns
    
    def calculate_dividend_yield(self):
        """Calculate dividend yield"""
        div_yield = self.info.get('dividendYield', 0)
        trailing_div = self.info.get('trailingAnnualDividendYield', 0)
        yield_val = div_yield or trailing_div
        return round(yield_val * 100, 2) if yield_val else 0.0
    

    from scipy import stats

    def calculate_risk_bucket(self):
        """
        ADVANCED RISK CALCULATION - Real Stock Market Metrics
        Uses multiple sophisticated risk measures used by professional traders
        """
        risk_score = 0
        risk_flags = []
        max_score = 100
        
        if self.hist is None or len(self.hist) < 30:
            return 'Medium'  # Not enough data
        
        # Calculate daily returns
        returns = self.hist['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return 'Medium'
        
        # ==================== 1. VOLATILITY ANALYSIS (25 points) ====================
        # Annualized volatility
        annual_vol = returns.std() * np.sqrt(252) * 100
        
        # Rolling volatility trend (increasing = more risky)
        rolling_vol = returns.rolling(30).std()
        vol_trend = (rolling_vol.iloc[-1] - rolling_vol.iloc[-30]) / rolling_vol.iloc[-30] if len(rolling_vol) > 30 else 0
        
        vol_score = 0
        if annual_vol > 50:  # Very high volatility
            vol_score = 25
            risk_flags.append(f"Vol:{annual_vol:.1f}%")
        elif annual_vol > 40:
            vol_score = 20
            risk_flags.append(f"Vol:{annual_vol:.1f}%")
        elif annual_vol > 30:
            vol_score = 15
        elif annual_vol > 20:
            vol_score = 10
        else:
            vol_score = 5
        
        # Add extra points if volatility is increasing
        if vol_trend > 0.3:
            vol_score += 5
            risk_flags.append("Vol↑")
        
        risk_score += vol_score
        
        # ==================== 2. VALUE AT RISK (VaR) - 20 points ====================
        # 95% VaR - Maximum expected loss in worst 5% of days
        var_95 = np.percentile(returns, 5) * 100
        
        # Conditional VaR (CVaR) - Average loss beyond VaR
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        var_score = 0
        if var_95 < -8:  # Extreme daily losses possible
            var_score = 20
            risk_flags.append(f"VaR:{var_95:.1f}%")
        elif var_95 < -6:
            var_score = 16
            risk_flags.append(f"VaR:{var_95:.1f}%")
        elif var_95 < -4:
            var_score = 12
        elif var_95 < -3:
            var_score = 8
        else:
            var_score = 4
        
        risk_score += var_score
        
        # ==================== 3. DRAWDOWN ANALYSIS (20 points) ====================
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration - how long stuck in drawdown
        in_drawdown = (drawdown < -5).astype(int)
        if in_drawdown.iloc[-1] == 1:
            dd_duration = (in_drawdown[::-1] != 0).cumsum()[in_drawdown[::-1] != 0].max()
        else:
            dd_duration = 0
        
        dd_score = 0
        if max_drawdown < -60:
            dd_score = 20
            risk_flags.append(f"DD:{max_drawdown:.1f}%")
        elif max_drawdown < -50:
            dd_score = 17
            risk_flags.append(f"DD:{max_drawdown:.1f}%")
        elif max_drawdown < -40:
            dd_score = 14
        elif max_drawdown < -30:
            dd_score = 11
        elif max_drawdown < -20:
            dd_score = 8
        else:
            dd_score = 4
        
        # Add penalty for prolonged drawdown
        if dd_duration > 60:
            dd_score += 5
            risk_flags.append(f"DD_Days:{dd_duration}")
        
        risk_score += dd_score
        
        # ==================== 4. TAIL RISK - Kurtosis & Skewness (15 points) ====================
        # Kurtosis > 3 = fat tails (more extreme events)
        # Negative skew = more downside crashes
        
        kurtosis = stats.kurtosis(returns)
        skewness = stats.skew(returns)
        
        tail_score = 0
        
        # Kurtosis penalty
        if kurtosis > 10:
            tail_score += 8
            risk_flags.append(f"Kurt:{kurtosis:.1f}")
        elif kurtosis > 5:
            tail_score += 6
        elif kurtosis > 3:
            tail_score += 4
        else:
            tail_score += 2
        
        # Skewness penalty (negative skew = bad)
        if skewness < -1.5:
            tail_score += 7
            risk_flags.append(f"Skew:{skewness:.2f}")
        elif skewness < -1.0:
            tail_score += 5
        elif skewness < -0.5:
            tail_score += 3
        else:
            tail_score += 1
        
        risk_score += tail_score
        
        # ==================== 5. BETA & MARKET SENSITIVITY (10 points) ====================
        beta = self.info.get('beta', 1.0)
        
        beta_score = 0
        if beta and beta > 0:
            if beta > 2.0:
                beta_score = 10
                risk_flags.append(f"β:{beta:.2f}")
            elif beta > 1.5:
                beta_score = 8
            elif beta > 1.2:
                beta_score = 6
            elif beta > 1.0:
                beta_score = 4
            else:
                beta_score = 2
        else:
            beta_score = 5  # Unknown beta = medium risk
        
        risk_score += beta_score
        
        # ==================== 6. SHARP PRICE MOVEMENTS (10 points) ====================
        # Count extreme daily moves
        extreme_moves = (abs(returns) > 0.05).sum()  # Days with >5% move
        extreme_pct = (extreme_moves / len(returns)) * 100
        
        # Recent volatility spike
        recent_vol = returns.tail(20).std()
        historical_vol = returns.std()
        vol_spike = recent_vol / historical_vol if historical_vol > 0 else 1
        
        movement_score = 0
        if extreme_pct > 15:
            movement_score += 6
            risk_flags.append(f"Extreme:{extreme_pct:.0f}%")
        elif extreme_pct > 10:
            movement_score += 4
        else:
            movement_score += 2
        
        if vol_spike > 1.5:
            movement_score += 4
            risk_flags.append("Vol_Spike")
        elif vol_spike > 1.2:
            movement_score += 2
        
        risk_score += movement_score
        
        # ==================== FINAL RISK CLASSIFICATION ====================
        # Normalize score to 0-100
        risk_score = min(risk_score, max_score)
        
        # REALISTIC THRESHOLDS based on actual market behavior
        if risk_score >= 70:
            risk_level = 'High'
            print(f"   🔴 HIGH RISK: {risk_score}/100 | {', '.join(risk_flags[:4])}")
        elif risk_score >= 45:
            risk_level = 'Medium'
            if risk_flags:
                print(f"   🟡 MEDIUM RISK: {risk_score}/100 | {', '.join(risk_flags[:3])}")
        else:
            risk_level = 'Low'
        
        # Store detailed metrics for analysis
        self.risk_metrics = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'annual_volatility': annual_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'beta': beta,
            'kurtosis': kurtosis,
            'skewness': skewness,
            'extreme_days_pct': extreme_pct,
            'flags': risk_flags
        }
        
        return risk_level
    



    
    
    def analyze(self):
        """Complete analysis"""
        if not self.fetch_live_data():
            return None
        
        result = {}
        result.update(self.get_basic_info())
        result.update(self.calculate_volatility())
        result.update(self.get_ohlc_data())
        result.update(self.calculate_returns())
        result['dividend_yield'] = self.calculate_dividend_yield()
        result['risk_bucket'] = self.calculate_risk_bucket()
        result['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return result


# ============================================================================
# 🔄 MAIN PROCESSING
# ============================================================================

def process_all_stocks(stock_list):
    INDEX_MAP = {
        "NIFTY 50": "^NSEI",
        "NIFTY NEXT 50": "^NSMIDCP",
        "BANKNIFTY": "^NSEBANK",
    }

    results = []
    total = len(stock_list)
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING {total} STOCKS")
    print(f"{'='*80}\n")
    
    for i, stock in enumerate(stock_list, 1):
        symbol = stock['symbol'].upper()
        
        # GET SECTOR FROM HARDCODED DATABASE - NO MORE "OTHERS"!
        sector = get_sector_from_symbol(symbol)

        if symbol in INDEX_MAP:
            ticker_yf = INDEX_MAP[symbol]
        else:
            ticker_yf = f"{symbol}.NS"
        
        print(f"[{i}/{total}] {symbol:15} | {sector:20} ... ", end="", flush=True)
        
        try:
            analyzer = LiveStockAnalyzer(ticker_yf, sector)
            data = analyzer.analyze()
            
            if data:
                data['ticker'] = symbol
                data['ticker_yf'] = ticker_yf
                results.append(data)
                print("✅")
            else:
                print("⚠️  No data")
                
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            continue
        
        if i % 50 == 0:
            print(f"\n{'='*80}")
            print(f"Progress: {i}/{total} ({len(results)} successful)")
            print(f"{'='*80}\n")
            time.sleep(2)
    
    return pd.DataFrame(results)


# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("🎯 FIXED NSE STOCK FETCHER - NO MORE 'OTHERS'!")
    print(f"{'='*80}\n")
    
    stock_list = fetch_all_nse_stocks()
    df = process_all_stocks(stock_list)
    
    os.makedirs("stock_data", exist_ok=True)
    # output_file = "data/live_nse_stocks_final1.csv"
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data"

    output_file = DATA_DIR / "live_nse_stocks_final1.csv"

    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 File: {output_file}")
    print(f"📊 Stocks: {len(df)}")
    
    if len(df) > 0:
        print(f"\n{'='*80}")
        print("📊 SECTOR DISTRIBUTION")
        print(f"{'='*80}")
        print(df['sector'].value_counts().to_string())
        
        print(f"\n{'='*80}")
        print("🎲 RISK DISTRIBUTION")
        print(f"{'='*80}")
        print(df['risk_bucket'].value_counts().to_string())
        
        return_3y_count = df['return_3y'].notna().sum()
        print(f"\n📈 Stocks with 3Y return: {return_3y_count}/{len(df)}")
        
        high_risk = df[df['risk_bucket'] == 'High']
        if len(high_risk) > 0:
            print(f"\n{'='*80}")
            print(f"⚠️  HIGH RISK STOCKS: {len(high_risk)}")
            print(f"{'='*80}")
            print(high_risk[['ticker', 'sector', 'beta', 'volatility_90d']].head(10).to_string(index=False))