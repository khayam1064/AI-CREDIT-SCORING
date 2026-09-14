"""P12 — Compliance Gate (F2, F10, F15) | Gate only — not weighted"""
import numpy as np
import pandas as pd

PILLAR_NAME = "Compliance"

# High-risk occupations (PEP / sanctioned job categories)
HIGH_RISK_OCCUPATIONS = {
    "Politician", "Government Official", "Military Officer",
    "Money Changer", "Pawnbroker", "Casino Staff",
}

# High-risk regions
HIGH_RISK_REGIONS = {"High"}


def gate(df: pd.DataFrame) -> pd.Series:
    """
    Returns per-customer compliance status:
      'PASS'  — proceed to scoring
      'REFER' — manual review required
      'FAIL'  — hard decline (gate failure)
    """
    status = pd.Series("PASS", index=df.index)

    # ── FAIL conditions (hard decline) ───────────────────────────────────────
    # 1. Sanctions flag (AML)
    aml_cols = [c for c in ["any_aml_flag", "is_sanctioned", "has_aml_flag"] if c in df.columns]
    if aml_cols:
        fail_aml = df[aml_cols].fillna(False).astype(bool).any(axis=1)
        status = np.where(fail_aml, "FAIL", status)

    # 2. Fraud-confirmed emulator (device spoofing = potential impersonation)
    if "dev_emulator_detected" in df.columns:
        fail_emu = df["dev_emulator_detected"].fillna(False).astype(bool)
        status = np.where(fail_emu, "FAIL", status)

    # 3. Extreme cross-lender velocity (>3 in 30d = potential fraud ring)
    if "app_cross_lender_velocity_30d" in df.columns:
        fail_vel = df["app_cross_lender_velocity_30d"].fillna(0) > 3
        status = np.where(fail_vel, "FAIL", status)

    # ── REFER conditions (manual review) ─────────────────────────────────────
    # 1. High-risk region
    if "risk_region" in df.columns:
        refer_region = df["risk_region"].isin(HIGH_RISK_REGIONS)
        status = np.where((status == "PASS") & refer_region, "REFER", status)

    # 2. High-risk occupation
    if "occupation_hint" in df.columns:
        refer_occ = df["occupation_hint"].isin(HIGH_RISK_OCCUPATIONS)
        status = np.where((status == "PASS") & refer_occ, "REFER", status)

    # 3. Multiple same-device applications
    if "app_applications_same_device_7d" in df.columns:
        refer_vel = df["app_applications_same_device_7d"].fillna(1) > 3
        status = np.where((status == "PASS") & refer_vel, "REFER", status)

    # 4. VPN + new customer combo (suspicious)
    if "dev_vpn_detected" in df.columns and "is_existing_customer" in df.columns:
        vpn_new = df["dev_vpn_detected"].fillna(False) & (~df["is_existing_customer"].fillna(False))
        status = np.where((status == "PASS") & vpn_new, "REFER", status)

    return pd.Series(status, index=df.index)


def score(df: pd.DataFrame) -> pd.Series:
    """Numeric representation: PASS=100, REFER=50, FAIL=0"""
    compliance_status = gate(df)
    return pd.Series(
        np.where(compliance_status == "PASS", 100.0,
        np.where(compliance_status == "REFER", 50.0, 0.0)),
        index=df.index
    )
