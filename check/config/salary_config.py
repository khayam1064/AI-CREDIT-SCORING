"""
Enterprise Credit AI
Salary Configuration

Defines realistic salary generation rules for synthetic customers.
"""

# ==========================================================
# BASE MONTHLY SALARY (PKR)
# ==========================================================

OCCUPATION_BASE_SALARY = {

    # IT
    "Software Engineer": 120000,
    "Senior Software Engineer": 220000,
    "Data Scientist": 240000,
    "AI Engineer": 280000,
    "DevOps Engineer": 210000,
    "Cyber Security Analyst": 180000,
    "System Administrator": 110000,
    "QA Engineer": 95000,
    "UI UX Designer": 120000,
    "Business Analyst": 150000,

    # Banking
    "Bank Teller": 65000,
    "Relationship Manager": 140000,
    "Credit Analyst": 160000,
    "Risk Manager": 260000,
    "Branch Manager": 320000,

    # Healthcare
    "Doctor": 250000,
    "Surgeon": 600000,
    "Nurse": 90000,
    "Pharmacist": 120000,

    # Education
    "School Teacher": 70000,
    "Lecturer": 130000,
    "Professor": 240000,

    # Government
    "Government Officer": 90000,
    "Police Officer": 85000,
    "Army Officer": 150000,

    # Business
    "Accountant": 120000,
    "Auditor": 145000,
    "Marketing Manager": 170000,
    "HR Manager": 165000,
    "Sales Manager": 180000,

    # Skilled
    "Electrician": 65000,
    "Plumber": 60000,
    "Mechanic": 70000,

    # Transport
    "Driver": 50000,
    "Truck Driver": 70000,
    "Pilot": 550000,

    # Retail
    "Cashier": 45000,
    "Sales Executive": 60000,
    "Store Manager": 100000,

    # Hospitality
    "Chef": 90000,
    "Hotel Manager": 180000,

    # Construction
    "Civil Engineer": 150000,
    "Architect": 180000,

    # Misc
    "Lawyer": 260000,
    "Consultant": 320000,
    "Entrepreneur": 250000
}

# ==========================================================
# INDUSTRY MULTIPLIERS
# ==========================================================

INDUSTRY_MULTIPLIER = {

    "IT": 1.45,
    "Banking": 1.35,
    "Healthcare": 1.30,
    "Telecom": 1.25,
    "Energy": 1.30,
    "Oil & Gas": 1.50,

    "Education": 0.95,
    "Government": 1.05,
    "Retail": 0.85,
    "Construction": 1.00,
    "Hospitality": 0.80,
    "Manufacturing": 0.95,
    "Transport": 0.90,
    "Agriculture": 0.75
}

# ==========================================================
# COMPANY SIZE
# ==========================================================

COMPANY_SIZE_MULTIPLIER = {

    "Startup": 0.90,
    "Small": 1.00,
    "Medium": 1.10,
    "Large": 1.20,
    "Enterprise": 1.35
}

# ==========================================================
# EDUCATION
# ==========================================================

EDUCATION_MULTIPLIER = {

    "Matric": 0.70,
    "Intermediate": 0.80,
    "Bachelor": 1.00,
    "Master": 1.15,
    "PhD": 1.30
}

# ==========================================================
# EMPLOYMENT TYPE
# ==========================================================

EMPLOYMENT_TYPE_MULTIPLIER = {

    "Permanent": 1.00,
    "Contract": 0.95,
    "Part-Time": 0.60,
    "Self-Employed": 1.10,
    "Business Owner": 1.30
}

# ==========================================================
# EXPERIENCE
# ==========================================================

EXPERIENCE_MULTIPLIER = {

    (0,2):0.75,
    (3,5):0.90,
    (6,10):1.10,
    (11,15):1.25,
    (16,20):1.40,
    (21,30):1.60,
    (31,40):1.75
}

# ==========================================================
# BONUS
# ==========================================================

BONUS_PERCENT = {

    "Low":0.05,
    "Medium":0.10,
    "High":0.20,
    "Executive":0.35
}

# ==========================================================
# CITY COST OF LIVING
# ==========================================================

CITY_MULTIPLIER = {

    "Karachi":1.15,
    "Lahore":1.10,
    "Islamabad":1.18,
    "Rawalpindi":1.05,
    "Faisalabad":1.00,
    "Multan":0.95,
    "Peshawar":0.95,
    "Hyderabad":0.92,
    "Quetta":0.90,
    "Sialkot":1.02,
    "Gujranwala":0.98,
    "Bahawalpur":0.90
}

# ==========================================================
# TAX BRACKETS
# ==========================================================

TAX_RATE = [

    (0,50000,0.00),

    (50001,100000,0.05),

    (100001,200000,0.10),

    (200001,350000,0.15),

    (350001,600000,0.20),

    (600001,99999999,0.30)
]

# ==========================================================
# PAYROLL
# ==========================================================

PAYROLL_DEPOSIT_PROBABILITY = {

    "Permanent":0.98,

    "Contract":0.90,

    "Part-Time":0.65,

    "Self-Employed":0.25,

    "Business Owner":0.15
}

# ==========================================================
# INFLATION
# ==========================================================

INFLATION_FACTOR = 1.00