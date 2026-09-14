"""
Enterprise Credit AI Engine
Realistic Synthetic Data Generator

Generates production-grade synthetic customer data with realistic 
credit scoring distributions following professional underwriting standards.

Target Distribution:
- 60% Approved, 25% Declined, 15% Refer
- Realistic financial ratios and segment archetypes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

class RealisticCreditDataGenerator:
    def __init__(self, num_records: int = 250000, random_seed: int = 42):
        """
        Initialize the realistic data generator.
        
        Args:
            num_records: Total number of customer records to generate
            random_seed: Random seed for reproducibility
        """
        self.num_records = num_records
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Target segment distributions
        self.segment_weights = {
            'prime_salaried': 0.35,      # 35% - Prime Salaried (Tier 1)
            'mid_tier': 0.30,            # 30% - Mid-Tier Salaried / Small Business  
            'gig_economy': 0.20,         # 20% - Gig Economy / Volatile Earners
            'over_indetted': 0.15        # 15% - Over-Indebted / Distressed
        }
        
        # Employment type distribution
        self.employment_types = {
            'salaried_corporate': 0.45,    # 45%
            'salaried_sme': 0.25,          # 25%
            'self_employed': 0.20,         # 20%
            'freelance_gig': 0.10          # 10%
        }
        
        # Tax filing distribution
        self.tax_filer_rate = 0.60  # 60% verified tax filers
        
        # Fraud flag distribution
        self.fraud_rate = 0.02  # 2% suspicious/triggered
        
    def generate_customer_id(self, index: int) -> str:
        """Generate normalized customer ID."""
        return f"CUST{index + 1:07d}"
    
    def assign_segment(self) -> str:
        """Assign customer segment based on target weights."""
        segments = list(self.segment_weights.keys())
        weights = list(self.segment_weights.values())
        return np.random.choice(segments, p=weights)
    
    def generate_demographics(self, segment: str) -> dict:
        """Generate realistic demographic data based on segment."""
        
        # Age distribution by segment
        if segment == 'prime_salaried':
            age = np.random.normal(35, 8)  # 35 ± 8 years
        elif segment == 'mid_tier':
            age = np.random.normal(32, 10)  # 32 ± 10 years
        elif segment == 'gig_economy':
            age = np.random.normal(28, 7)   # 28 ± 7 years
        else:  # over_indetted
            age = np.random.normal(40, 12)  # 40 ± 12 years
        
        age = max(21, min(65, int(age)))  # Clamp to realistic range
        
        # Gender distribution
        gender = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        
        # Education by segment
        if segment == 'prime_salaried':
            education = np.random.choice(['Masters', 'Bachelor', 'PhD'], p=[0.4, 0.5, 0.1])
        elif segment == 'mid_tier':
            education = np.random.choice(['Bachelor', 'Masters', 'High School'], p=[0.6, 0.3, 0.1])
        elif segment == 'gig_economy':
            education = np.random.choice(['Bachelor', 'High School', 'Masters'], p=[0.4, 0.4, 0.2])
        else:
            education = np.random.choice(['High School', 'Bachelor', 'Masters'], p=[0.5, 0.4, 0.1])
        
        # Marital status
        marital_status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.5, 0.4, 0.1])
        
        # Location (Pakistan cities)
        cities = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 
                  'Multan', 'Peshawar', 'Quetta', 'Sialkot', 'Gujranwala']
        city = np.random.choice(cities)
        
        # Province based on city
        province_map = {
            'Karachi': 'Sindh', 'Lahore': 'Punjab', 'Islamabad': 'ICT',
            'Rawalpindi': 'Punjab', 'Faisalabad': 'Punjab', 'Multan': 'Punjab',
            'Peshawar': 'KPK', 'Quetta': 'Balochistan', 'Sialkot': 'Punjab',
            'Gujranwala': 'Punjab'
        }
        province = province_map.get(city, 'Punjab')
        
        # Residential status
        if segment == 'prime_salaried':
            residential_status = np.random.choice(['Own', 'Rent', 'Family'], p=[0.6, 0.3, 0.1])
        elif segment == 'over_indetted':
            residential_status = np.random.choice(['Rent', 'Own', 'Family'], p=[0.6, 0.3, 0.1])
        else:
            residential_status = np.random.choice(['Rent', 'Own', 'Family'], p=[0.5, 0.3, 0.2])
        
        return {
            'age': age,
            'gender': gender,
            'education': education,
            'marital_status': marital_status,
            'city': city,
            'province': province,
            'residential_status': residential_status
        }
    
    def generate_employment(self, segment: str) -> dict:
        """Generate employment information based on segment."""
        
        # Employment type based on segment
        if segment == 'prime_salaried':
            emp_type = np.random.choice(['salaried_corporate', 'salaried_sme'], p=[0.8, 0.2])
        elif segment == 'mid_tier':
            emp_type = np.random.choice(['salaried_sme', 'salaried_corporate', 'self_employed'], p=[0.5, 0.3, 0.2])
        elif segment == 'gig_economy':
            emp_type = np.random.choice(['freelance_gig', 'self_employed', 'salaried_sme'], p=[0.6, 0.3, 0.1])
        else:  # over_indetted
            emp_type = np.random.choice(['salaried_sme', 'self_employed', 'freelance_gig'], p=[0.4, 0.4, 0.2])
        
        # Employment status
        employment_status = 'Employed' if emp_type != 'freelance_gig' else np.random.choice(['Employed', 'Self-Employed'])
        
        # Industry
        industries = ['Banking & Finance', 'Technology', 'Manufacturing', 'Retail', 
                      'Healthcare', 'Education', 'Construction', 'Transportation',
                      'Agriculture', 'Services']
        industry = np.random.choice(industries)
        
        # Company size
        if emp_type == 'salaried_corporate':
            company_size = np.random.choice(['Large', 'Medium', 'Small'], p=[0.6, 0.3, 0.1])
        else:
            company_size = np.random.choice(['Small', 'Medium', 'Large'], p=[0.6, 0.3, 0.1])
        
        # Manager level
        if segment == 'prime_salaried':
            manager_level = np.random.choice(['Senior', 'Mid', 'Junior'], p=[0.3, 0.5, 0.2])
        else:
            manager_level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.5, 0.4, 0.1])
        
        return {
            'employment_status': employment_status,
            'employment_type': emp_type,
            'industry': industry,
            'company_size': company_size,
            'manager_level': manager_level
        }
    
    def generate_income_and_financials(self, segment: str) -> dict:
        """Generate income and financial data based on segment characteristics."""
        
        # Income ranges by segment (PKR)
        if segment == 'prime_salaried':
            base_income = np.random.uniform(120000, 450000)
        elif segment == 'mid_tier':
            base_income = np.random.uniform(50000, 120000)
        elif segment == 'gig_economy':
            base_income = np.random.uniform(25000, 150000)
        else:  # over_indetted
            base_income = np.random.uniform(30000, 200000)
        
        # Add some randomness
        income_variation = np.random.normal(1.0, 0.15)  # ±15% variation
        total_monthly_income = max(20000, base_income * income_variation)
        
        # Salary components
        gross_monthly_salary = total_monthly_income * np.random.uniform(0.7, 0.9)
        net_monthly_salary = gross_monthly_salary * np.random.uniform(0.75, 0.85)
        basic_salary = gross_monthly_salary * np.random.uniform(0.6, 0.8)
        
        # FOIR (Fixed Obligation to Income Ratio) - key for approval
        if segment == 'prime_salaried':
            # Low debt: < 30% FOIR (45% of this segment)
            foir = np.random.uniform(0.15, 0.30)
        elif segment == 'mid_tier':
            # Moderate debt: 30-50% FOIR (mixed)
            foir = np.random.uniform(0.30, 0.50)
        elif segment == 'gig_economy':
            # Variable debt, can be high
            foir = np.random.uniform(0.25, 0.60)
        else:  # over_indetted
            # High debt: > 50% FOIR (20% excessive)
            foir = np.random.uniform(0.50, 0.75)
        
        # Calculate existing obligations based on FOIR
        existing_monthly_obligations = total_monthly_income * foir
        
        # Balance-to-Income Ratio
        if segment == 'prime_salaried':
            # Healthy cushion: 1.0-4.0x (60% of records)
            balance_to_income = np.random.uniform(1.0, 4.0)
        elif segment == 'mid_tier':
            # Mixed: 0.05-4.0x
            balance_to_income = np.random.choice(
                [np.random.uniform(0.05, 0.5), np.random.uniform(1.0, 4.0)],
                p=[0.4, 0.6]
            )
        elif segment == 'gig_economy':
            # Tight: 0.05-0.5x (30% paycheck-to-paycheck)
            balance_to_income = np.random.uniform(0.05, 0.8)
        else:  # over_indetted
            # Very tight or anomalous
            balance_to_income = np.random.choice(
                [np.random.uniform(0.01, 0.3), np.random.uniform(0.5, 1.5)],
                p=[0.7, 0.3]
            )
        
        avg_monthly_balance = total_monthly_income * balance_to_income
        
        # Credit-to-Debit Flow Ratio
        if segment == 'prime_salaried':
            # Net accumulator: ≥1.15x (50%)
            credit_debit_ratio = np.random.uniform(1.15, 1.8)
        elif segment == 'mid_tier':
            # Balanced: 0.95-1.14x (35%)
            credit_debit_ratio = np.random.uniform(0.95, 1.14)
        elif segment == 'gig_economy':
            # Capital deficit: <0.95x (15%)
            credit_debit_ratio = np.random.uniform(0.70, 1.10)
        else:  # over_indetted
            # Often living off debt
            credit_debit_ratio = np.random.uniform(0.60, 1.20)
        
        # Calculate total credit and debit amounts (12M)
        # Assume average monthly turnover based on income
        avg_monthly_turnover = total_monthly_income * np.random.uniform(1.5, 3.0)
        total_annual_turnover = avg_monthly_turnover * 12
        
        # Split based on credit-debit ratio
        total_credit_amt_12m = total_annual_turnover * (credit_debit_ratio / (1 + credit_debit_ratio))
        total_debit_amt_12m = total_annual_turnover / (1 + credit_debit_ratio)
        
        # Monthly averages
        avg_monthly_debit_flow = total_debit_amt_12m / 12
        
        # Utility payments (portion of debit)
        utility_debit_amt_12m = total_debit_amt_12m * np.random.uniform(0.15, 0.35)
        
        # Tax amount (based on filer status)
        is_tax_filer = np.random.random() < self.tax_filer_rate
        if is_tax_filer:
            tax_amount = total_monthly_income * np.random.uniform(0.02, 0.08) * 12  # Annual tax
        else:
            tax_amount = 0.0
        
        # Overdraft limit and usage
        if segment == 'over_indetted':
            overdraft_limit = total_monthly_income * np.random.uniform(0.5, 1.5)
            overdraft_usage = np.random.uniform(0.70, 0.95)  # High usage
        elif segment == 'prime_salaried':
            overdraft_limit = total_monthly_income * np.random.uniform(0.3, 0.8)
            overdraft_usage = np.random.uniform(0.10, 0.40)  # Low usage
        else:
            overdraft_limit = total_monthly_income * np.random.uniform(0.3, 1.0)
            overdraft_usage = np.random.uniform(0.20, 0.60)
        
        total_overdraft_limit = overdraft_limit
        overdraft_utilization = overdraft_usage
        
        # Account tenure
        if segment == 'prime_salaried':
            account_tenure_months = np.random.randint(24, 120)  # 2-10 years
        elif segment == 'gig_economy':
            account_tenure_months = np.random.randint(6, 48)  # 6 months - 4 years
        else:
            account_tenure_months = np.random.randint(12, 72)  # 1-6 years
        
        # Number of accounts
        num_accounts = np.random.randint(1, 4)
        
        # Income stability score
        if segment == 'prime_salaried':
            # High stability: ≥75% (55% of records)
            income_stability_score = np.random.uniform(75, 95)
        elif segment == 'mid_tier':
            # Moderate: 40-74% (25%)
            income_stability_score = np.random.uniform(40, 80)
        elif segment == 'gig_economy':
            # High volatility: <40% (20%)
            income_stability_score = np.random.uniform(20, 50)
        else:  # over_indetted
            # Variable
            income_stability_score = np.random.uniform(30, 70)
        
        # Income confidence score
        income_confidence_score = income_stability_score * np.random.uniform(0.9, 1.1)
        income_confidence_score = min(100, max(0, income_confidence_score))
        
        # Fraud flag
        has_fraud_flag = np.random.random() < self.fraud_rate
        
        # Verified income
        verified_income = gross_monthly_salary if is_tax_filer else gross_monthly_salary * 0.8
        
        return {
            'total_monthly_income': total_monthly_income,
            'gross_monthly_salary': gross_monthly_salary,
            'net_monthly_salary': net_monthly_salary,
            'basic_salary': basic_salary,
            'verified_income': verified_income,
            'avg_monthly_balance': avg_monthly_balance,
            'balance_to_income_ratio': balance_to_income,
            'total_credit_amt_12m': total_credit_amt_12m,
            'total_debit_amt_12m': total_debit_amt_12m,
            'avg_monthly_debit_flow': avg_monthly_debit_flow,
            'utility_debit_amt_12m': utility_debit_amt_12m,
            'tax_amount': tax_amount,
            'total_overdraft_limit': total_overdraft_limit,
            'overdraft_utilization': overdraft_utilization,
            'account_tenure_months': account_tenure_months,
            'number_of_accounts': num_accounts,
            'income_stability_score': income_stability_score,
            'income_confidence_score': income_confidence_score,
            'has_fraud_flag': has_fraud_flag
        }
    
    def generate_customer_record(self, index: int) -> dict:
        """Generate a complete customer record."""
        
        # Assign segment
        segment = self.assign_segment()
        
        # Generate all components
        customer_id = self.generate_customer_id(index)
        demographics = self.generate_demographics(segment)
        employment = self.generate_employment(segment)
        financials = self.generate_income_and_financials(segment)
        
        # Combine all data
        record = {
            'customer_id': customer_id,
            'segment': segment,
            **demographics,
            **employment,
            **financials
        }
        
        return record
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        
        print(f"Generating {self.num_records:,} realistic customer records...")
        print("Segment distribution:")
        for segment, weight in self.segment_weights.items():
            expected_count = int(self.num_records * weight)
            print(f"  {segment}: {weight*100:.1f}% (~{expected_count:,} records)")
        
        records = []
        for i in range(self.num_records):
            if (i + 1) % 50000 == 0:
                print(f"  Generated {i + 1:,} records...")
            
            record = self.generate_customer_record(i)
            records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Dataset generation complete: {len(df):,} records")
        
        # Print actual segment distribution
        print("\nActual segment distribution:")
        print(df['segment'].value_counts(normalize=True).mul(100).round(1))
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: Path):
        """Save dataset to parquet file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        # Print summary statistics
        print("\nDataset Summary Statistics:")
        print(f"Total records: {len(df):,}")
        print(f"Total monthly income range: PKR {df['total_monthly_income'].min():,.0f} - PKR {df['total_monthly_income'].max():,.0f}")
        print(f"Average monthly income: PKR {df['total_monthly_income'].mean():,.0f}")
        print(f"Average FOIR: {(df['avg_monthly_debit_flow'] / df['total_monthly_income']).mean()*100:.1f}%")
        print(f"Fraud flag rate: {(df['has_fraud_flag'].sum() / len(df) * 100):.2f}%")
        print(f"Tax filer rate: {((df['tax_amount'] > 0).sum() / len(df) * 100):.1f}%")

if __name__ == "__main__":
    # Configuration
    NUM_RECORDS = 250000
    OUTPUT_PATH = Path(__file__).parent.parent / "feature_store" / "customer_features.parquet"
    
    # Generate dataset
    generator = RealisticCreditDataGenerator(num_records=NUM_RECORDS, random_seed=42)
    df = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(df, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("Realistic dataset generation complete!")
    print("Expected model performance: ~60% Approve / 40% Decline+Refer")
    print("="*60)