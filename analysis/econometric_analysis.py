#!/usr/bin/env python3
"""
analysis/econometric_analysis.py
=================================
Econometric Identification Engine for ABM-Earth-TropEcS.
Estimates cross-sectional and country-clustered two-way panel econometric models
quantifying structural drivers of:
1. Unmonitored Carrier Transshipment & Dark-Fleet Loitering Exposure (z_log_loiter)
2. Foreign Distant-Water Fleet (DWF) Penetration into Coastal EEZs (z_dwf)

Specifications Estimated (Carrier/Dark-Fleet Loitering Suite):
- Model 1: Bivariate OLS (Standardized, Maritime-Realm-Clustered SEs, N = 109)
- Model 2: Multivariate OLS with Log GDP per capita & Capacity Subsidies (HC1 Robust SEs, N = 109)
- Model 3: Maritime Realm Fixed Effects (HC1 Robust SEs, N = 109)
- Model 4: Extended Policy Specification (MPA Share & Trophic Asymmetry Index, HC1 Robust SEs, N = 109)
- Model 5a: Longitudinal Panel Realm + Year Fixed Effects (2012-2024, N = 1,417, Country-Clustered SEs)
- Model 5: Longitudinal Two-Way Fixed Effects (Country + Year FE, 2012-2024, N = 1,417, Country-Clustered SEs)

Diagnostics Computed:
- Variance Inflation Factors (VIF) for Multicollinearity
- Sovereign Country-Clustered Standard Errors across all 109 coastal states in panel regressions

Exports:
- tables/Table_Econometric_Panel_Regressions.csv
- tables/Table_Econometric_Panel_Regressions.md
- tables/Table_DWF_Exposure_Regressions.csv
- tables/Table_VIF_Multicollinearity.csv
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EconometricAnalysis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std())
    if np.isnan(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Computes transformed and standardized covariates consistently across cross-section and panel."""
    out = df.copy()
    gdp_med = float(out["gdp_pc"].replace(0, np.nan).median())
    out["log_gdp"] = np.log(out["gdp_pc"].replace(0, np.nan)).fillna(np.log(gdp_med))
    out["log_subsidies"] = np.log(out["capacity_subsidies_usd"].clip(lower=0.0) + 1.0)
    out["dwf_share"] = out["foreign_dwf_share"].fillna(out["foreign_dwf_share"].median())
    out["log_loiter"] = np.log1p(out["carrier_loitering_hours"].clip(lower=0.0))
    out["mpa_pct"] = out["wb_mpa_pct_territorial"].fillna(out["wb_mpa_pct_territorial"].median())
    out["trophic_asym"] = out["trophic_asymmetry_index"].fillna(0.0)

    out["z_log_loiter"] = _zscore(out["log_loiter"])
    out["z_dwf"] = _zscore(out["dwf_share"])
    out["z_log_gdp"] = _zscore(out["log_gdp"])
    out["z_log_sub"] = _zscore(out["log_subsidies"])
    out["z_mpa"] = _zscore(out["mpa_pct"])
    out["z_asym"] = _zscore(out["trophic_asym"])
    return out


def load_and_prep_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads cross-sectional profiles (N = 109) and GFW satellite-era longitudinal panel (N = 1,417)."""
    profiles_path = DATA_DIR / "country_profiles.csv"
    panel_path = DATA_DIR / "empirical_panel.parquet"

    if not profiles_path.exists():
        raise FileNotFoundError(f"Country profiles not found at {profiles_path}")

    profiles = _prepare_covariates(pd.read_csv(profiles_path))

    panel = None
    if panel_path.exists():
        panel_raw = pd.read_parquet(panel_path)
        panel_sub = panel_raw[
            (panel_raw["country_iso3"].isin(profiles["country_iso3"])) & (panel_raw["year"] >= 2012)
        ].copy()
        panel = _prepare_covariates(panel_sub)

    return profiles, panel


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Computes Variance Inflation Factors for multicollinearity evaluation."""
    X = sm.add_constant(df[feature_cols].dropna())
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    vif_data["Collinearity_Risk"] = vif_data["VIF"].apply(
        lambda x: "Low (< 3.0)" if x < 3.0 else ("Moderate (3.0 - 5.0)" if x < 5.0 else "High (> 5.0)")
    )
    return vif_data


def _compile_regression_table(
    models: Dict[str, object],
    se_labels: Dict[str, str],
    key_vars: List[Tuple[str, List[str]]]
) -> pd.DataFrame:
    """Formats fitted statsmodels OLS regressions into a publication summary table."""
    rows = []
    for label, param_keys in key_vars:
        row = {"Variable": label}
        for m_name, model in models.items():
            val_str = "—"
            for pk in param_keys:
                if pk in model.params:
                    coef = float(model.params[pk])
                    se = float(model.bse[pk])
                    pval = float(model.pvalues[pk])
                    stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                    val_str = f"{coef:.3f}{stars} ({se:.3f})"
                    break
            row[m_name] = val_str
        rows.append(row)

    obs_row = {"Variable": "Observations (N)"}
    r2_row = {"Variable": "R-squared"}
    r2_adj_row = {"Variable": "Adjusted R-squared"}
    fe_ctry_row = {"Variable": "Country Fixed Effects"}
    fe_realm_row = {"Variable": "Marine Realm Fixed Effects"}
    fe_year_row = {"Variable": "Year Fixed Effects"}
    se_type_row = {"Variable": "Standard Error Clustering"}

    for m_name, model in models.items():
        obs_row[m_name] = str(int(model.nobs))
        r2_row[m_name] = f"{model.rsquared:.3f}"
        r2_adj_row[m_name] = f"{model.rsquared_adj:.3f}"
        fe_ctry_row[m_name] = "Yes" if m_name == "Model 5 (Panel Two-Way FE)" else "No"
        fe_realm_row[m_name] = "Yes" if m_name in ["Model 3 (Realm FE)", "Model 4 (Extended Policy)", "Model 5a (Panel Realm + Year FE)"] else "No"
        fe_year_row[m_name] = "Yes" if "Panel" in m_name else "No"
        se_type_row[m_name] = se_labels.get(m_name, "Robust")

    rows.extend([obs_row, r2_row, r2_adj_row, fe_ctry_row, fe_realm_row, fe_year_row, se_type_row])
    return pd.DataFrame(rows)


def _fit_suite(profiles: pd.DataFrame, panel: pd.DataFrame, dep_var: str, include_dwf: bool) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Fits the 6-specification cross-sectional and country-clustered panel suite."""
    rhs_m2 = "z_dwf + z_log_gdp + z_log_sub" if include_dwf else "z_log_gdp + z_log_sub"
    rhs_m4 = "z_dwf + z_log_gdp + z_log_sub + z_mpa + z_asym" if include_dwf else "z_log_gdp + z_log_sub + z_mpa + z_asym"
    rhs_p = "z_dwf + z_log_gdp + z_log_sub + z_mpa" if include_dwf else "z_log_gdp + z_log_sub + z_mpa"
    bivar_x = "z_dwf" if include_dwf else "z_log_gdp"

    models = {
        "Model 1 (Bivariate)": smf.ols(f"{dep_var} ~ {bivar_x}", data=profiles).fit(
            cov_type="cluster", cov_kwds={"groups": profiles["region_classification"]}
        ),
        "Model 2 (Multivariate)": smf.ols(f"{dep_var} ~ {rhs_m2}", data=profiles).fit(cov_type="HC1"),
        "Model 3 (Realm FE)": smf.ols(
            f"{dep_var} ~ {rhs_m2} + C(region_classification)", data=profiles
        ).fit(cov_type="HC1"),
        "Model 4 (Extended Policy)": smf.ols(
            f"{dep_var} ~ {rhs_m4} + C(region_classification)", data=profiles
        ).fit(cov_type="HC1"),
    }
    se_labels = {
        "Model 1 (Bivariate)": "Maritime Realm Cluster",
        "Model 2 (Multivariate)": "Heteroskedasticity (HC1)",
        "Model 3 (Realm FE)": "Heteroskedasticity (HC1)",
        "Model 4 (Extended Policy)": "Heteroskedasticity (HC1)",
    }

    if panel is not None:
        models["Model 5a (Panel Realm + Year FE)"] = smf.ols(
            f"{dep_var} ~ {rhs_p} + C(region_classification) + C(year)", data=panel
        ).fit(cov_type="cluster", cov_kwds={"groups": panel["country_iso3"]})
        models["Model 5 (Panel Two-Way FE)"] = smf.ols(
            f"{dep_var} ~ {rhs_p} + C(country_iso3) + C(year)", data=panel
        ).fit(cov_type="cluster", cov_kwds={"groups": panel["country_iso3"]})
        se_labels["Model 5a (Panel Realm + Year FE)"] = "Country-Clustered (109 States)"
        se_labels["Model 5 (Panel Two-Way FE)"] = "Country-Clustered (109 States)"

    return models, se_labels


def run_econometric_suite():
    """Estimates all specifications, saves summary tables, and computes VIF diagnostics."""
    profiles, panel = load_and_prep_data()
    logger.info(f"Loaded {len(profiles)} country profiles for econometric estimation.")

    key_vars_loiter = [
        ("DWF Effort Share", ["z_dwf"]),
        ("Log GDP per capita", ["z_log_gdp"]),
        ("Log Capacity Subsidies", ["z_log_sub"]),
        ("Marine Protected Area (%)", ["z_mpa"]),
        ("Trophic Asymmetry Index", ["z_asym"]),
        ("Intercept", ["Intercept"]),
    ]

    # Primary suite: Unmonitored Carrier & Dark-Fleet Loitering Exposure
    models_loiter, se_loiter = _fit_suite(profiles, panel, dep_var="z_log_loiter", include_dwf=True)
    df_reg = _compile_regression_table(models_loiter, se_loiter, key_vars_loiter)

    out_csv = TABLES_DIR / "Table_Econometric_Panel_Regressions.csv"
    out_md = TABLES_DIR / "Table_Econometric_Panel_Regressions.md"
    df_reg.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("**Extended Data Table 2 | Econometric Drivers of Unmonitored Carrier Loitering & Dark-Fleet Activity Across 109 Sovereign Coastal States**\n\n")
        f.write(df_reg.to_markdown(index=False))
        f.write("\n\n*Notes: Dependent variable is standardized log unmonitored carrier & dark-vessel loitering hours (`z_log_loiter`). Standardized beta coefficients reported with standard errors in parentheses. *** p < 0.001, ** p < 0.01, * p < 0.05 (two-sided t/Wald tests). Model 1 uses maritime-realm-clustered standard errors; Models 2–4 report HC1 robust standard errors across 109 sovereign coastal profiles; Models 5a and 5 report country-clustered standard errors across 109 states over the 2012–2024 GFW satellite panel (N = 1,417).*\n")

    # Companion suite: Foreign Distant-Water Fleet Penetration vs. Coastal Income & Subsidies
    key_vars_dwf = [k for k in key_vars_loiter if k[0] != "DWF Effort Share"]
    models_dwf, se_dwf = _fit_suite(profiles, panel, dep_var="z_dwf", include_dwf=False)
    df_dwf = _compile_regression_table(models_dwf, se_dwf, key_vars_dwf)
    out_dwf_csv = TABLES_DIR / "Table_DWF_Exposure_Regressions.csv"
    df_dwf.to_csv(out_dwf_csv, index=False)

    # VIF diagnostics
    feature_cols = ["dwf_share", "log_gdp", "log_subsidies", "mpa_pct", "trophic_asym"]
    vif_df = compute_vif(profiles, feature_cols)
    out_vif = TABLES_DIR / "Table_VIF_Multicollinearity.csv"
    vif_df.to_csv(out_vif, index=False)

    logger.info(f"Saved primary regressions to {out_csv}, companion DWF suite to {out_dwf_csv}, and VIF to {out_vif}")

    print("\n" + "=" * 88)
    print("ECONOMETRIC SUITE: UNMONITORED CARRIER & DARK-FLEET LOITERING (z_log_loiter)")
    print("=" * 88)
    print(df_reg.to_string(index=False))
    print("\n" + "-" * 88)
    print("COMPANION SUITE: FOREIGN DWF EFFORT SHARE PENETRATION (z_dwf)")
    print("-" * 88)
    print(df_dwf.to_string(index=False))
    print("\n" + "-" * 88)
    print("VARIANCE INFLATION FACTORS (VIF) MULTICOLLINEARITY DIAGNOSTICS")
    print("-" * 88)
    print(vif_df.to_string(index=False))
    print("=" * 88 + "\n")

    return df_reg, vif_df


if __name__ == "__main__":
    run_econometric_suite()
