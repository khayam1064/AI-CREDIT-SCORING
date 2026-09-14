"""
Enterprise Credit AI Engine
Master Customer Data Generator & Validation Pipeline

Generates 100% compliant synthetic customer records serving as the master entity
store for downstream employment, income, banking, transaction, and credit engines.
"""

import os
import re
import sys
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Base Path Config
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "raw"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOMER_FILE = DATASETS_DIR / "customers.csv"

# Global Constants & Seeds
TOTAL_CUSTOMERS = 250000
RANDOM_SEED = 42
TODAY = datetime(2026, 8, 5)

fake = Faker()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Business Configurations & Lookup Tables
# -----------------------------------------------------------------------------

MALE_FIRST_NAMES = [
    "Muhammad", "Ali", "Ahmed", "Usman", "Hamza", "Bilal", "Umer", "Zain", 
    "Tariq", "Hassan", "Hussein", "Omer", "Mustafa", "Asad", "Kamran", "Fahad",
    "Shahzaib", "Haris", "Saad", "Imran", "Salman", "Aamir", "Waqas", "Nabeel"
]

FEMALE_FIRST_NAMES = [
    "Fatima", "Ayesha", "Zainab", "Maryam", "Sana", "Amna", "Hira", "Iqra",
    "Sara", "Sadia", "Anum", "Nida", "Mahnoor", "Laiba", "Fariha", "Sidra",
    "Khadija", "Noreen", "Bushra", "Mehwish", "Samreen", "Rabia", "Tehreem"
]

LAST_NAMES = [
    "Khan", "Ahmed", "Malik", "Chaudhry", "Bhatti", "Qureshi", "Sheikh", 
    "Syed", "Raza", "Hussain", "Shah", "Mirza", "Farooqi", "Abbasi", "Mughal",
    "Javed", "Iqbal", "Siddiqui", "Mahmood", "Tariq", "Akram", "Ghaffar"
]

GENDER_DIST = {"Male": 0.68, "Female": 0.32}

AGE_GROUPS = [
    ((18, 25), 0.25),
    ((26, 35), 0.40),
    ((36, 50), 0.22),
    ((51, 65), 0.10),
    ((66, 75), 0.03)
]

EDUCATION_MIN_AGE = {
    "Primary": 6,
    "Secondary": 14,
    "Intermediate": 16,
    "Bachelor": 20,
    "Master": 22,
    "PhD": 26
}

CITY_DATA = {
    "Karachi": {"province": "Sindh", "postal_code": "74000", "risk_region": "Medium"},
    "Lahore": {"province": "Punjab", "postal_code": "54000", "risk_region": "Low"},
    "Islamabad": {"province": "Islamabad Capital Territory", "postal_code": "44000", "risk_region": "Low"},
    "Rawalpindi": {"province": "Punjab", "postal_code": "46000", "risk_region": "Low"},
    "Faisalabad": {"province": "Punjab", "postal_code": "38000", "risk_region": "Medium"},
    "Peshawar": {"province": "Khyber Pakhtunkhwa", "postal_code": "25000", "risk_region": "High"},
    "Quetta": {"province": "Balochistan", "postal_code": "87300", "risk_region": "High"},
    "Multan": {"province": "Punjab", "postal_code": "60000", "risk_region": "Medium"},
    "Sialkot": {"province": "Punjab", "postal_code": "51310", "risk_region": "Low"},
    "Hyderabad": {"province": "Sindh", "postal_code": "71000", "risk_region": "Medium"}
}

CITY_DIST = {
    "Karachi": 0.30, "Lahore": 0.25, "Islamabad": 0.12, "Rawalpindi": 0.08,
    "Faisalabad": 0.07, "Peshawar": 0.05, "Multan": 0.04, "Sialkot": 0.04,
    "Hyderabad": 0.03, "Quetta": 0.02
}

MOBILE_PREFIX_OPERATOR = {
    "0300": "Jazz", "0301": "Jazz", "0302": "Jazz", "0303": "Jazz", "0304": "Jazz", "0305": "Jazz",
    "0310": "Zong", "0311": "Zong", "0312": "Zong", "0313": "Zong", "0314": "Zong",
    "0320": "Warid", "0321": "Warid", "0322": "Warid",
    "0330": "Ufone", "0331": "Ufone", "0332": "Ufone", "0333": "Ufone",
    "0340": "Telenor", "0341": "Telenor", "0342": "Telenor", "0345": "Telenor"
}

EMAIL_PROVIDERS = {"gmail.com": 0.65, "outlook.com": 0.15, "yahoo.com": 0.12, "hotmail.com": 0.08}
RESIDENTIAL_STATUS_DIST = {"Owned": 0.45, "Rental": 0.35, "Family": 0.15, "Government": 0.03, "Hostel": 0.02}
CUSTOMER_SEGMENT_DIST = {"Mass": 0.60, "Affluent": 0.25, "Premium": 0.10, "Private Banking": 0.05}
OCCUPATION_HINTS = ["Student", "Engineer", "Doctor", "Teacher", "Business", "Retired", "Homemaker"]
PHONE_TYPES = {"Prepaid": 0.80, "Postpaid": 0.20}
LANGUAGES = {"Urdu": 0.70, "English": 0.30}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def weighted_choice(mapping):
    choices = list(mapping.keys())
    weights = list(mapping.values())
    return random.choices(choices, weights=weights, k=1)[0]

def random_date_between(start_date, end_date):
    if start_date >= end_date:
        return start_date
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, max(0, delta.days)))

# Global sets for strict uniqueness
generated_cnics = set()
generated_phones = set()
generated_emails = set()

def generate_cnic_unique(gender):
    while True:
        area = random.randint(11111, 55555)
        serial = random.randint(1000000, 9999999)
        gender_digit = random.choice([1, 3, 5, 7, 9]) if gender == "Male" else random.choice([2, 4, 6, 8])
        cnic = f"{area}-{serial}-{gender_digit}"
        if cnic not in generated_cnics:
            generated_cnics.add(cnic)
            return cnic

def generate_phone_unique():
    while True:
        prefix = random.choice(list(MOBILE_PREFIX_OPERATOR.keys()))
        suffix = "".join([str(random.randint(0, 9)) for _ in range(7)])
        phone = f"{prefix}{suffix}"
        if phone not in generated_phones:
            generated_phones.add(phone)
            return phone, prefix, MOBILE_PREFIX_OPERATOR[prefix]

def generate_email_unique(first_name, last_name, customer_id):
    while True:
        domain = weighted_choice(EMAIL_PROVIDERS)
        clean_first = re.sub(r'[^a-zA-Z]', '', first_name.lower())
        clean_last = re.sub(r'[^a-zA-Z]', '', last_name.lower())
        clean_cid = re.sub(r'[^0-9]', '', customer_id)
        email = f"{clean_first}.{clean_last}.{clean_cid}@{domain}"
        if email not in generated_emails:
            generated_emails.add(email)
            return email, domain

# -----------------------------------------------------------------------------
# Core Generator Functions
# -----------------------------------------------------------------------------

def generate_customer_record(cid_num):
    customer_id = f"CUST{cid_num:07d}"
    
    # 1. Gender & Name
    gender = weighted_choice(GENDER_DIST)
    first_name = random.choice(MALE_FIRST_NAMES if gender == "Male" else FEMALE_FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    
    # 2. Age & DOB Construction (Guaranteed Synchronized)
    cnic = generate_cnic_unique(gender)
    group = random.choices(AGE_GROUPS, weights=[x[1] for x in AGE_GROUPS], k=1)[0]
    age = random.randint(group[0][0], group[0][1])
    
    # Pick DOB such that (TODAY - DOB) in exact calendar years equals 'age'
    # Month/day picked before today's month/day ensures birthday has occurred this year
    # Pick random month and day safely before or on TODAY's date to avoid age boundary offsets
    birth_year = TODAY.year - age
    month = random.randint(1, 7) # Up to July ensures birthday already passed in 2026
    day = random.randint(1, 28)
    dob = datetime(birth_year, month, day)

    # 3. Education Timeline Logic
    eligible_education = [ed for ed, min_a in EDUCATION_MIN_AGE.items() if age >= min_a]
    education = random.choice(eligible_education)

    # 4. Marital Status Logic
    if age < 18:
        marital_status = "Single"
    elif age < 24:
        marital_status = weighted_choice({"Single": 0.85, "Married": 0.15})
    elif age < 35:
        marital_status = weighted_choice({"Single": 0.35, "Married": 0.62, "Divorced": 0.03})
    else:
        marital_status = weighted_choice({"Married": 0.75, "Single": 0.12, "Divorced": 0.08, "Widowed": 0.05})

    # 5. Occupation Hint Logic
    if age < 22 and education in ["Primary", "Secondary", "Intermediate"]:
        occupation_hint = "Student"
    elif age >= 60:
        occupation_hint = weighted_choice({"Retired": 0.70, "Business": 0.20, "Homemaker": 0.10})
    else:
        occupation_hint = weighted_choice({
            "Engineer": 0.25, "Doctor": 0.15, "Teacher": 0.20,
            "Business": 0.25, "Homemaker": 0.15
        })

    # 6. Exact Timeline Sequencing
    birth_date = dob
    tenth_bday = datetime(birth_year + 10, month, day)
    fifteenth_bday = datetime(birth_year + 15, month, day)
    eighteenth_bday = datetime(birth_year + 18, month, day)

    # Email Creation (> 10th Birthday & < TODAY)
    email_created_date = random_date_between(tenth_bday, TODAY - timedelta(days=30))
    email_age_days = (TODAY - email_created_date).days
    email, email_provider = generate_email_unique(first_name, last_name, customer_id)

    # SIM Activation (> 15th Birthday & > Email Creation & < TODAY)
    earliest_sim = max(fifteenth_bday, email_created_date)
    sim_activation_date = random_date_between(earliest_sim, TODAY - timedelta(days=10))
    sim_age_days = (TODAY - sim_activation_date).days
    phone_number, mobile_prefix, mobile_operator = generate_phone_unique()
    phone_type = weighted_choice(PHONE_TYPES)

    # Address Move-in Date (>= 18th Birthday & < TODAY)
    move_in_date = random_date_between(eighteenth_bday, TODAY - timedelta(days=5))
    years_at_address = round((TODAY - move_in_date).days / 365.25, 2)

    # Customer Since (>= 18th Birthday & >= Move In & < TODAY)
    is_existing_customer = random.choices([True, False], weights=[0.65, 0.35], k=1)[0]
    if is_existing_customer:
        earliest_customer_since = max(eighteenth_bday, move_in_date)
        customer_since = random_date_between(earliest_customer_since, TODAY - timedelta(days=1))
    else:
        customer_since = None

    # 7. Geographic Attributes
    city = weighted_choice(CITY_DIST)
    city_info = CITY_DATA[city]
    province = city_info["province"]
    postal_code = city_info["postal_code"]
    risk_region = city_info["risk_region"]

    # 8. Demographic Metadata
    residential_status = weighted_choice(RESIDENTIAL_STATUS_DIST)
    customer_segment = weighted_choice(CUSTOMER_SEGMENT_DIST)
    preferred_language = weighted_choice(LANGUAGES)
    device_count = random.randint(1, 6)

    return {
        "customer_id": customer_id,
        "cnic": cnic,
        "first_name": first_name,
        "last_name": last_name,
        "gender": gender,
        "age": age,
        "dob": dob.strftime("%Y-%m-%d"),
        "marital_status": marital_status,
        "education": education,
        "occupation_hint": occupation_hint,
        "country": "Pakistan",
        "nationality": "Pakistani",
        "city": city,
        "province": province,
        "postal_code": postal_code,
        "risk_region": risk_region,
        "phone_number": phone_number,
        "mobile_prefix": mobile_prefix,
        "mobile_operator": mobile_operator,
        "phone_type": phone_type,
        "sim_activation_date": sim_activation_date.strftime("%Y-%m-%d"),
        "sim_age_days": sim_age_days,
        "email": email,
        "email_provider": email_provider,
        "email_created_date": email_created_date.strftime("%Y-%m-%d"),
        "email_age_days": email_age_days,
        "residential_status": residential_status,
        "move_in_date": move_in_date.strftime("%Y-%m-%d"),
        "years_at_address": years_at_address,
        "is_existing_customer": is_existing_customer,
        "customer_since": customer_since.strftime("%Y-%m-%d") if customer_since else None,
        "preferred_language": preferred_language,
        "device_count": device_count,
        "customer_segment": customer_segment
    }

def generate_dataset():
    logging.info(f"Generating {TOTAL_CUSTOMERS:,} enterprise customer records...")
    records = [generate_customer_record(cid) for cid in tqdm(range(1, TOTAL_CUSTOMERS + 1))]
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# Enterprise 36-Point Validation Pipeline
# -----------------------------------------------------------------------------

def run_enterprise_validation(df):
    logging.info("Executing Enterprise 36-Point Data Quality Audit...")
    results = {}
    
    # Primary Key Validation
    results["Customer ID Format Check"] = df["customer_id"].str.match(r"^CUST\d{7}$").all()
    results["Customer ID Null Check"] = df["customer_id"].isnull().sum() == 0
    results["Zero Duplicate Customer IDs"] = df["customer_id"].duplicated().sum() == 0
    
    # CNIC Validation
    cnic_regex = r"^\d{5}-\d{7}-\d$"
    results["CNIC Format Check"] = df["cnic"].str.match(cnic_regex).all()
    results["Zero Duplicate CNICs"] = df["cnic"].duplicated().sum() == 0
    results["CNIC Null Check"] = df["cnic"].isnull().sum() == 0

    # Name & Gender Consistency
    results["First Name Alphabetic"] = df["first_name"].str.isalpha().all()
    results["Last Name Alphabetic"] = df["last_name"].str.isalpha().all()
    male_name_valid = df[df["gender"] == "Male"]["first_name"].isin(MALE_FIRST_NAMES).all()
    female_name_valid = df[df["gender"] == "Female"]["first_name"].isin(FEMALE_FIRST_NAMES).all()
    results["Gender-Name Alignment Check"] = male_name_valid and female_name_valid

    # Age and DOB Checks
    df["dob_dt"] = pd.to_datetime(df["dob"])
    calc_ages = df["dob_dt"].apply(lambda d: TODAY.year - d.year - ((TODAY.month, TODAY.day) < (d.month, d.day)))
    results["Age Range Check (18-75)"] = df["age"].between(18, 75).all()
    results["Age Matches DOB Calculation"] = (df["age"] - calc_ages).abs().max() == 0
    results["No Future Birthdays"] = (df["dob_dt"] < TODAY).all()

    # Marital Status Rules
    under_18_married = df[(df["age"] < 18) & (df["marital_status"] != "Single")]
    results["Marital Status Age Logic"] = len(under_18_married) == 0

    # Education Rules
    ed_errors = sum(len(df[(df["education"] == ed) & (df["age"] < min_a)]) for ed, min_a in EDUCATION_MIN_AGE.items())
    results["Education Age Thresholds"] = ed_errors == 0

    # Location & Geographic Checks
    results["City Existence Check"] = df["city"].isin(CITY_DATA.keys()).all()
    results["Province-City Alignment"] = df.apply(lambda r: CITY_DATA[r["city"]]["province"] == r["province"], axis=1).all()
    results["Postal Code Alignment"] = df.apply(lambda r: CITY_DATA[r["city"]]["postal_code"] == r["postal_code"], axis=1).all()
    results["Country Check (Pakistan)"] = (df["country"] == "Pakistan").all()
    results["Nationality Check (Pakistani)"] = (df["nationality"] == "Pakistani").all()

    # Phone & Operator Checks
    results["Phone Format Check (^03)"] = df["phone_number"].str.match(r"^03\d{9}$").all()
    results["Zero Duplicate Phones"] = df["phone_number"].duplicated().sum() == 0
    results["Mobile Operator Mapping"] = df.apply(lambda r: MOBILE_PREFIX_OPERATOR[r["mobile_prefix"]] == r["mobile_operator"], axis=1).all()

    # SIM Timeline & Age
    df["sim_activation_dt"] = pd.to_datetime(df["sim_activation_date"])
    fifteenth_bdays = df["dob_dt"] + pd.DateOffset(years=15)
    results["SIM Activation Timeline (>15th Bday)"] = (df["sim_activation_dt"] >= fifteenth_bdays).all()
    calc_sim_age = (TODAY - df["sim_activation_dt"]).dt.days
    results["SIM Age Computation Match"] = (df["sim_age_days"] == calc_sim_age).all()

    # Email Checks
    results["Email Format Regex Check"] = df["email"].str.contains(r"^[\w\.-]+@[\w\.-]+\.\w+$").all()
    results["Email Provider Validity"] = df["email_provider"].isin(EMAIL_PROVIDERS.keys()).all()
    results["Zero Duplicate Emails"] = df["email"].duplicated().sum() == 0
    df["email_created_dt"] = pd.to_datetime(df["email_created_date"])
    tenth_bdays = df["dob_dt"] + pd.DateOffset(years=10)
    results["Email Creation Timeline (>10th Bday)"] = (df["email_created_dt"] >= tenth_bdays).all() and (df["email_created_dt"] < TODAY).all()
    calc_email_age = (TODAY - df["email_created_dt"]).dt.days
    results["Email Age Computation Match"] = (df["email_age_days"] == calc_email_age).all()

    # Address Tenure Checks
    results["Residence Type Validity"] = df["residential_status"].isin(RESIDENTIAL_STATUS_DIST.keys()).all()
    df["move_in_dt"] = pd.to_datetime(df["move_in_date"])
    eighteenth_bdays = df["dob_dt"] + pd.DateOffset(years=18)
    results["Move-in Date Timeline (>18th Bday)"] = (df["move_in_dt"] >= eighteenth_bdays).all()
    calc_address_years = ((TODAY - df["move_in_dt"]).dt.days / 365.25).round(2)
    results["Address Tenure Match"] = (df["years_at_address"] - calc_address_years).abs().max() <= 0.05

    # Existing Customer Logic Checks
    non_exist_null = df[df["is_existing_customer"] == False]["customer_since"].isnull().all()
    exist_not_null = df[df["is_existing_customer"] == True]["customer_since"].notnull().all()
    results["Existing Customer Logic Flags"] = non_exist_null and exist_not_null

    df["customer_since_dt"] = pd.to_datetime(df["customer_since"])
    valid_cs_dates = df[df["is_existing_customer"] == True]
    cs_eighteenth_bdays = valid_cs_dates["dob_dt"] + pd.DateOffset(years=18)
    cs_after_18 = (valid_cs_dates["customer_since_dt"] >= cs_eighteenth_bdays).all()
    cs_before_today = (valid_cs_dates["customer_since_dt"] <= TODAY).all()
    results["Customer Since Timeline Validity"] = cs_after_18 and cs_before_today

    # Metadata & Rules
    results["Language Support"] = df["preferred_language"].isin(LANGUAGES.keys()).all()
    results["Risk Region Alignment"] = df.apply(lambda r: CITY_DATA[r["city"]]["risk_region"] == r["risk_region"], axis=1).all()
    results["Device Count Bounds (1-8)"] = df["device_count"].between(1, 8).all()
    results["Occupation Hint Validity"] = df["occupation_hint"].isin(OCCUPATION_HINTS).all()
    results["Customer Segment Validity"] = df["customer_segment"].isin(CUSTOMER_SEGMENT_DIST.keys()).all()

    # Master Timeline Sequencing
    results["Master Timeline Chronological Order"] = (
        (df["dob_dt"] < df["email_created_dt"]).all() and
        (df["email_created_dt"] <= df["sim_activation_dt"]).all() and
        (df["sim_activation_dt"] <= TODAY).all()
    )

    # Missing & Datatypes
    required_cols = [
        "customer_id", "cnic", "first_name", "last_name", "dob", "age",
        "gender", "phone_number", "email", "city", "province"
    ]
    results["Required Fields Missing Values"] = df[required_cols].isnull().sum().sum() == 0
    results["Data Types & Numeric Bounds"] = (df["age"].dtype == np.int64 or df["age"].dtype == int)
    
    # Distribution Check
    gender_p = df["gender"].value_counts(normalize=True)["Male"]
    results["Gender Distribution Match"] = abs(gender_p - GENDER_DIST["Male"]) <= 0.05

    # Cleanup temp columns
    df.drop(columns=[c for c in df.columns if c.endswith("_dt")], inplace=True)
    return results

def print_validation_report(results, df):
    total_checks = len(results)
    passed_checks = sum(results.values())
    quality_score = (passed_checks / total_checks) * 100

    report = f"""
============================================================
        CUSTOMER DATA QUALITY REPORT
============================================================

Records Generated              : {len(df):,}

Schema Validation              : {"PASS" if results["Customer ID Format Check"] else "FAIL"}
Customer ID                    : {"PASS" if results["Customer ID Null Check"] else "FAIL"}
CNIC Validation                : {"PASS" if results["CNIC Format Check"] else "FAIL"}
Duplicate Customer IDs         : {"PASS" if results["Zero Duplicate Customer IDs"] else "FAIL"}
Duplicate CNIC                 : {"PASS" if results["Zero Duplicate CNICs"] else "FAIL"}
Duplicate Emails               : {"PASS" if results["Zero Duplicate Emails"] else "FAIL"}
Duplicate Phones               : {"PASS" if results["Zero Duplicate Phones"] else "FAIL"}

Age Validation                 : {"PASS" if results["Age Range Check (18-75)"] else "FAIL"}
DOB Validation                 : {"PASS" if results["No Future Birthdays"] else "FAIL"}
Gender Validation              : {"PASS" if results["Gender-Name Alignment Check"] else "FAIL"}
Education Validation           : {"PASS" if results["Education Age Thresholds"] else "FAIL"}
Marital Status Validation      : {"PASS" if results["Marital Status Age Logic"] else "FAIL"}

City Mapping                   : {"PASS" if results["City Existence Check"] else "FAIL"}
Province Mapping               : {"PASS" if results["Province-City Alignment"] else "FAIL"}
Postal Code Validation         : {"PASS" if results["Postal Code Alignment"] else "FAIL"}

Phone Validation               : {"PASS" if results["Phone Format Check (^03)"] else "FAIL"}
Operator Mapping               : {"PASS" if results["Mobile Operator Mapping"] else "FAIL"}
SIM Timeline                   : {"PASS" if results["SIM Activation Timeline (>15th Bday)"] else "FAIL"}

Email Format                   : {"PASS" if results["Email Format Regex Check"] else "FAIL"}
Email Timeline                 : {"PASS" if results["Email Creation Timeline (>10th Bday)"] else "FAIL"}

Residence Validation           : {"PASS" if results["Residence Type Validity"] else "FAIL"}
Address Timeline               : {"PASS" if results["Move-in Date Timeline (>18th Bday)"] else "FAIL"}

Existing Customer Logic        : {"PASS" if results["Existing Customer Logic Flags"] else "FAIL"}
Customer Since Validation      : {"PASS" if results["Customer Since Timeline Validity"] else "FAIL"}

Missing Required Values        : {"PASS" if results["Required Fields Missing Values"] else "FAIL"}
Data Types                     : {"PASS" if results["Data Types & Numeric Bounds"] else "FAIL"}
Business Rules                 : {"PASS" if results["Master Timeline Chronological Order"] else "FAIL"}
Probability Distribution       : {"PASS" if results["Gender Distribution Match"] else "FAIL"}

Overall Data Quality Score     : {quality_score:.2f}%
Dataset Status                 : {"PRODUCTION READY" if quality_score == 100.0 else "VALIDATION FAILED"}
============================================================
"""
    print(report)
    if quality_score < 100.0:
        failed_tests = [k for k, v in results.items() if not v]
        print("FAILED CHECKS:", failed_tests)
        raise Exception("Dataset Quality Assurance Failed.")

def save_dataset(df):
    df.to_csv(CUSTOMER_FILE, index=False, encoding="utf-8")
    logging.info(f"Master customer dataset successfully saved to: {CUSTOMER_FILE}")

def main():
    df = generate_dataset()
    results = run_enterprise_validation(df)
    print_validation_report(results, df)
    save_dataset(df)

if __name__ == "__main__":
    main()