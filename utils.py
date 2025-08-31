\
import numpy as np
import pandas as pd
from math import exp
from sklearn.preprocessing import StandardScaler

Q_REF = 120.0  # nominal flow m3/h

def compute_viscosity(temp_c: float) -> float:
    \"\"\"Estimate dynamic viscosity (mPa*s) vs. temperature.
    mu(T) = mu0 * exp(k*(10 - T)); mu0 ~ 2 mPa*s at 10C, k ~ 0.06
    \"\"\"
    if pd.isna(temp_c):
        return np.nan
    return 2.0 * np.exp(0.06 * (10.0 - temp_c))

def cavitation_risk_proxy(row: pd.Series) -> float:
    \"\"\"Cold-aware cavitation risk proxy (dimensionless).
    Uses: Pressure_Fluct_bar (if present), viscosity index, and flow reduction.
    \"\"\"
    pf = float(row.get("Pressure_Fluct_bar", 0) or 0)
    T = row.get("Ambient_T", np.nan)
    mu = row.get("Viscosity_mPa_s", np.nan)
    if pd.isna(mu) and not pd.isna(T):
        mu = compute_viscosity(T)
    visc_idx = 0.0 if pd.isna(mu) else (mu / 2.0)  # ~1 at 10C, >1 in cold
    Q = row.get("Flow_Rate", np.nan)
    flow_red = 0.0 if pd.isna(Q) else max(0.0, (Q_REF - float(Q)) / Q_REF)
    risk = pf * (0.6 + 0.4*visc_idx) * (1.0 + flow_red)
    return float(risk)

def cold_factor(temp_c: float) -> float:
    \"\"\"Map temperature to a 0..1 cold stress factor (<= -20C -> ~1).\"\"\"
    if pd.isna(temp_c):
        return 0.0
    x = -float(temp_c)  # colder -> larger
    s = 1.0 / (1.0 + np.exp(-(x - 10.0)/4.0))
    return float(s)

def risk_score(df_row: pd.Series, proba_normal: float = None) -> float:
    \"\"\"Compute overall risk 0..1 using available signals + model confidence.\"\"\"
    vrms = float(df_row.get("Vibration_RMS_g", 0) or 0)
    v1x  = float(df_row.get("Vibration_1x_g", 0) or 0)
    pfl  = float(df_row.get("Pressure_Fluct_bar", 0) or 0)
    leak = float(df_row.get("Leakage_Rate", 0) or 0)
    temp = df_row.get("Ambient_T", np.nan)
    mu   = df_row.get("Viscosity_mPa_s", np.nan)
    if pd.isna(mu) and not pd.isna(temp):
        mu = compute_viscosity(temp)
    cavp = cavitation_risk_proxy(df_row)
    cold = cold_factor(temp) if not pd.isna(temp) else 0.0

    vr = min(vrms/2.0, 1.0)
    v1 = min(v1x/1.5, 1.0)
    pf = min(pfl/0.2, 1.0)
    lk = min(leak/2.0, 1.0)
    mu_idx = 0.0 if pd.isna(mu) else min(mu/6.0, 1.0)
    cv = min(cavp/0.5, 1.0)

    model_risk = 0.0 if (proba_normal is None) else (1.0 - proba_normal)

    score = (0.23*vr + 0.12*v1 + 0.18*pf + 0.12*lk + 0.10*mu_idx + 0.15*cv + 0.10*cold + 0.10*model_risk)
    return float(max(0.0, min(1.0, score)))

def classify_status(score: float, cur_label: str) -> str:
    \"\"\"Map to three statuses: Excellent / Needs Attention / Faulty\"\"\"
    if cur_label not in ("Normal", "", None):
        return "Faulty" if score >= 0.6 else "Needs Attention"
    if score >= 0.8:
        return "Faulty"
    elif score >= 0.55:
        return "Needs Attention"
    else:
        return "Excellent"
