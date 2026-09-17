from pydantic import BaseModel, Field
from typing import Optional

class RealtimeApplicantPayload(BaseModel):
    customer_id: Optional[str] = 'APPLICANT_REALTIME'
    age: Optional[int] = Field(default=35, ge=18, le=100)
    gender: Optional[str] = 'Male'
    marital_status: Optional[str] = 'Single'
    education: Optional[str] = 'Bachelor'
    city: Optional[str] = 'Islamabad'
    province: Optional[str] = 'Punjab'
    residential_status: Optional[str] = 'Rent'
    years_at_address: Optional[float] = 5.0
    
    # Employment & Income
    employment_status: Optional[str] = 'Employed'
    industry: Optional[str] = 'Banking & Finance'
    company_size: Optional[str] = 'Medium'
    years_of_experience: Optional[float] = 5.0
    current_job_tenure: Optional[float] = 2.0
    monthly_salary: Optional[float] = 120000.0
    total_monthly_income: Optional[float] = 120000.0
    net_monthly_cashflow: Optional[float] = 30000.0
    foir_pct: Optional[float] = 15.0
    
    # Device & Behavior
    dev_os_type: Optional[str] = 'Android'
    dev_os_version: Optional[int] = 13
    dev_is_rooted: Optional[bool] = False
    dev_emulator_detected: Optional[bool] = False
    dev_typing_speed_wpm: Optional[float] = 25.0
    dev_hesitation_score: Optional[float] = 20.0
    
    # Telco & Wallet
    tel_sim_age_months: Optional[int] = 60
    tel_is_postpaid: Optional[bool] = False
    wal_has_wallet: Optional[bool] = True
    wal_wallet_provider: Optional[str] = 'JazzCash'
    wal_avg_monthly_txn_value: Optional[float] = 25000.0
    
    # Ciihive LMS & History
    ldr_has_internal_history: Optional[bool] = False
    ldr_prior_loans_count: Optional[int] = 0
    ldr_active_loans_count: Optional[int] = 0
    ldr_on_time_payment_rate: Optional[float] = 1.0
    ldr_avg_dpd_last_12m: Optional[float] = 0.0
    savings_ratio: Optional[float] = 0.10
    
    # Compliance Knockouts
    is_pep: Optional[bool] = False
    is_sanctioned: Optional[bool] = False
    has_aml_flag: Optional[bool] = False
