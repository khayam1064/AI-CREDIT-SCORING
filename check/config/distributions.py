"""
Enterprise Credit AI
Distribution Configuration

This file contains all probability distributions used by
synthetic dataset generators.
"""

# ============================================
# NUMBER OF CUSTOMERS
# ============================================

TOTAL_CUSTOMERS = 250_000

# ============================================
# AGE DISTRIBUTION
# ============================================

AGE_GROUPS = [
    ((18, 22), 0.08),
    ((23, 30), 0.28),
    ((31, 40), 0.35),
    ((41, 50), 0.20),
    ((51, 60), 0.09),
]

# ============================================
# GENDER
# ============================================

GENDER = {
    "Male": 0.52,
    "Female": 0.48
}

# ============================================
# EDUCATION
# ============================================

EDUCATION = {
    "Matric": 0.18,
    "Intermediate": 0.22,
    "Bachelor": 0.38,
    "Master": 0.18,
    "PhD": 0.04
}

# ============================================
# CUSTOMER SEGMENT
# ============================================

CUSTOMER_SEGMENT = {
    "Mass": 0.72,
    "Affluent": 0.23,
    "Premium": 0.05
}

# ============================================
# RESIDENTIAL STATUS
# ============================================

RESIDENTIAL_STATUS = {
    "Own": 0.42,
    "Rent": 0.45,
    "Family": 0.13
}

# ============================================
# PHONE TYPE
# ============================================

PHONE_TYPE = {
    "Prepaid": 0.65,
    "Postpaid": 0.35
}

# ============================================
# EMAIL PROVIDERS
# ============================================

EMAIL_PROVIDER = {
    "gmail.com": 0.55,
    "outlook.com": 0.15,
    "yahoo.com": 0.10,
    "icloud.com": 0.07,
    "proton.me": 0.03,
    "company.com": 0.10
}

# ============================================
# NATIONALITY
# ============================================

NATIONALITY = {
    "Pakistani": 0.97,
    "Foreign Resident": 0.03
}

# ============================================
# CITIES
# ============================================

CITY = {
    "Karachi": 0.22,
    "Lahore": 0.18,
    "Islamabad": 0.08,
    "Rawalpindi": 0.08,
    "Faisalabad": 0.10,
    "Peshawar": 0.08,
    "Multan": 0.07,
    "Hyderabad": 0.06,
    "Quetta": 0.05,
    "Sialkot": 0.03,
    "Gujranwala": 0.03,
    "Bahawalpur": 0.02
}

# ============================================
# CITY → PROVINCE
# ============================================

CITY_PROVINCE = {
    "Karachi": "Sindh",
    "Hyderabad": "Sindh",

    "Lahore": "Punjab",
    "Rawalpindi": "Punjab",
    "Faisalabad": "Punjab",
    "Multan": "Punjab",
    "Sialkot": "Punjab",
    "Gujranwala": "Punjab",
    "Bahawalpur": "Punjab",

    "Islamabad": "ICT",

    "Peshawar": "KPK",

    "Quetta": "Balochistan"
}