#!/usr/bin/env python3
"""
global_sensitivity_analysis.py
Dense Sobol Global Sensitivity Analysis (GSA) using 30-Year ML Surrogate Emulation,
paired with direct-ABM Saltelli rank-concordance validation.
Generates Saltelli sequence sampling (N = 73,728 evaluations across D=17 parameters).
Computes:
- First-order indices (S1): Main effects
- Total-order indices (ST): Total effects including all non-linear interactions
- Second-order indices (S2): Pairwise parameter synergies
- Direct-ABM vs. Surrogate Spearman rank concordance across all 17 parameters (N = 304 ABM runs)

Exports:
- tables/Table_Sobol_Sensitivity_Indices.csv
- tables/Table_Sobol_Second_Order_S2.csv
- tables/Table_Direct_ABM_Sobol_Validation.csv
- tables/sobol_results.parquet
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent
TARGET_DIR = ANALYSIS_DIR.parent
TABLES_DIR = TARGET_DIR / "tables"
MODEL_DIR = TARGET_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from engine import ABMEarthEngine

try:
    from .surrogate_emulator import OUTPUT_METRICS, PARAMETER_SPEC, SurrogateEmulator
except ImportError:
    from surrogate_emulator import OUTPUT_METRICS, PARAMETER_SPEC, SurrogateEmulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GlobalSensitivityAnalysis")


def validate_direct_abm_sobol(
    emulator: SurrogateEmulator,
    df_sobol: pd.DataFrame,
    n_base_direct: int = 16,
    duration_years: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Runs a compact first/total-order Saltelli design directly on ABMEarthEngine
    (N_direct = (D + 2) * n_base_direct = 304 full 30-year simulations) to verify
    that surrogate-derived Total-Order Sobol rankings match true ABM parameter ordering.
    """
    param_names = list(PARAMETER_SPEC.keys())
    problem_direct = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": [list(PARAMETER_SPEC[p]) for p in param_names]
    }
    X_direct = sobol_sample.sample(problem_direct, n_base_direct, calc_second_order=False, seed=seed)
    n_direct = len(X_direct)
    logger.info(f"Running {n_direct} direct 30-year ABM simulations for Sobol rank validation...")

    y_bio = np.zeros(n_direct, dtype=float)
    y_dt = np.zeros(n_direct, dtype=float)
    for idx, x_row in enumerate(X_direct):
        p = dict(zip(param_names, x_row))
        eng = ABMEarthEngine(
            scenario="SSP2-4.5",
            start_year=2020,
            end_year=2020 + duration_years,
            r_growth=float(p["r_growth"]),
            phi_crit=float(p["phi_crit"]),
            capacity_subsidy_rate=float(p["capacity_subsidy_rate"]),
            fuel_price_volatility=float(p["fuel_price_volatility"]),
            risk_aversion=float(p["risk_aversion"]),
            debt_threshold=float(p["debt_threshold"]),
            carbon_price=float(p["carbon_price"]),
            export_tariff_pct=float(p["export_tariff_pct"]),
            mpa_fraction=float(p["mpa_fraction"]),
            mangrove_fraction=float(p["mangrove_fraction"]),
            carrying_capacity_k=float(p["carrying_capacity_k"]),
            num_ssf_fishers=int(round(p["num_ssf_fishers"])),
            num_dwf_vessels=int(round(p["num_dwf_vessels"])),
            sst_warming_rate=float(p["sst_warming_rate"]),
            base_fuel_price=float(p["base_fuel_price"]),
            ex_vessel_price_base=float(p["ex_vessel_price_base"]),
            diffusivity=float(p["diffusivity"]),
            seed=seed,
        )
        df_run = eng.run_simulation()
        y_bio[idx] = float(df_run["total_biomass"].iloc[-1])
        y_dt[idx] = float(df_run["debt_trap_fraction"].max())

    rows = []
    for metric, y_vec in [("final_biomass", y_bio), ("peak_debt_trap_fraction", y_dt)]:
        res_dir = sobol_analyze.analyze(problem_direct, y_vec, calc_second_order=False, seed=seed)
        sub_surr = df_sobol[df_sobol["Metric"] == metric].set_index("Parameter")
        st_dir = np.maximum(0.0, res_dir["ST"])
        st_surr = sub_surr.loc[param_names, "Total_Order_ST"].values
        rho, pval = stats.spearmanr(st_dir, st_surr)
        logger.info(f"  Direct-ABM vs. Surrogate Sobol ST [{metric}]: Spearman rho = {rho:.3f} (p = {pval:.2e})")
        for idx_p, pname in enumerate(param_names):
            rows.append({
                "Metric": metric,
                "Parameter": pname,
                "Direct_ABM_S1": round(max(0.0, float(res_dir["S1"][idx_p])), 4),
                "Direct_ABM_ST": round(float(st_dir[idx_p]), 4),
                "Surrogate_ST": round(float(st_surr[idx_p]), 4),
                "Spearman_rho": round(float(rho), 4),
                "Spearman_p_value": f"{pval:.2e}",
                "Direct_N_Runs": n_direct,
            })

    df_val = pd.DataFrame(rows)
    out_val = TABLES_DIR / "Table_Direct_ABM_Sobol_Validation.csv"
    df_val.to_csv(out_val, index=False)
    logger.info(f"Saved direct-ABM Sobol validation table to {out_val}")
    return df_val


def run_sobol_analysis(n_base_samples: int = 2048, seed: int = 42):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = ANALYSIS_DIR / "surrogate_bundle.pkl"

    if not bundle_path.exists():
        logger.info("Surrogate bundle not found. Training 30-year surrogate first (N_train = 512)...")
        emulator = SurrogateEmulator(seed=seed)
        X_lhs = emulator.generate_lhs_samples(n_samples=512)
        df_train = emulator.run_training_simulations(X_lhs, duration_years=30)
        emulator.train_emulators(df_train)
        emulator.benchmark_architectures(df_train)
        emulator.save(bundle_path)
    else:
        logger.info(f"Loading existing surrogate emulator from {bundle_path}...")
        emulator = SurrogateEmulator.load(bundle_path)

    param_names = list(PARAMETER_SPEC.keys())
    bounds = [list(PARAMETER_SPEC[p]) for p in param_names]
    problem = {"num_vars": len(param_names), "names": param_names, "bounds": bounds}

    logger.info(f"Generating Saltelli sequence sample with base N={n_base_samples} across {len(param_names)} parameters...")
    param_values = sobol_sample.sample(problem, n_base_samples, calc_second_order=True, seed=seed)
    n_evals = len(param_values)
    logger.info(f"Total Saltelli evaluation points: {n_evals:,}")

    all_results = []
    s2_results = []

    for metric in OUTPUT_METRICS:
        logger.info(f"Evaluating surrogate predictions for metric: [{metric}] across {n_evals:,} points...")
        preds = []
        chunk_size = 50000
        for i in range(0, n_evals, chunk_size):
            preds.append(emulator.predict(param_values[i:i + chunk_size], metric))
        y = np.concatenate(preds)

        logger.info(f"Computing Sobol variance decomposition for [{metric}]...")
        sobol_indices = sobol_analyze.analyze(problem, y, calc_second_order=True, seed=seed)

        s1 = sobol_indices["S1"]
        s1_conf = sobol_indices["S1_conf"]
        st = sobol_indices["ST"]
        st_conf = sobol_indices["ST_conf"]
        s2_mat = sobol_indices["S2"]
        s2_conf_mat = sobol_indices["S2_conf"]

        for idx, p_name in enumerate(param_names):
            s1_val = max(0.0, float(s1[idx]))
            st_val = max(s1_val, float(st[idx]))
            s1_ci = round(float(s1_conf[idx]), 4)
            st_ci = round(float(st_conf[idx]), 4)
            all_results.append({
                "Parameter": p_name,
                "Metric": metric,
                "First_Order_S1": round(s1_val, 4),
                "S1_Confidence": s1_ci,
                "First_Order_S1_CI": s1_ci,
                "Total_Order_ST": round(st_val, 4),
                "ST_Confidence": st_ci,
                "Total_Order_ST_CI": st_ci,
                "Interaction_ST_minus_S1": round(max(0.0, st_val - s1_val), 4)
            })

        n_params = len(param_names)
        for i in range(n_params):
            for j in range(i + 1, n_params):
                val_s2 = float(s2_mat[i, j]) if not np.isnan(s2_mat[i, j]) else 0.0
                ci_s2 = float(s2_conf_mat[i, j]) if not np.isnan(s2_conf_mat[i, j]) else 0.0
                s2_results.append({
                    "Parameter_1": param_names[i],
                    "Parameter_2": param_names[j],
                    "Metric": metric,
                    "Second_Order_S2": round(max(0.0, val_s2), 4),
                    "S2_Confidence": round(abs(ci_s2), 4)
                })

    df_sobol = pd.DataFrame(all_results)
    df_s2 = pd.DataFrame(s2_results).sort_values(["Metric", "Second_Order_S2"], ascending=[True, False])

    csv_out = TABLES_DIR / "Table_Sobol_Sensitivity_Indices.csv"
    s2_csv_out = TABLES_DIR / "Table_Sobol_Second_Order_S2.csv"
    parquet_out = TABLES_DIR / "sobol_results.parquet"
    df_sobol.to_csv(csv_out, index=False)
    df_s2.to_csv(s2_csv_out, index=False)
    df_sobol.to_parquet(parquet_out, index=False)

    validate_direct_abm_sobol(emulator, df_sobol, n_base_direct=16, duration_years=30, seed=seed)

    logger.info(f"Sobol GSA completed! Saved results to {csv_out}, {s2_csv_out}, and {parquet_out}")

    print("\n" + "=" * 80)
    print("TOP SENSITIVITY DRIVERS: FINAL BIOMASS (Biomass_2050)")
    print("=" * 80)
    bio_sub = df_sobol[df_sobol["Metric"] == "final_biomass"].sort_values("Total_Order_ST", ascending=False)
    print(bio_sub[["Parameter", "First_Order_S1", "Total_Order_ST", "Interaction_ST_minus_S1"]].head(6).to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP SENSITIVITY DRIVERS: PEAK DEBT-TRAP FRACTION (DebtTrap_Max)")
    print("=" * 80)
    dt_sub = df_sobol[df_sobol["Metric"] == "peak_debt_trap_fraction"].sort_values("Total_Order_ST", ascending=False)
    print(dt_sub[["Parameter", "First_Order_S1", "Total_Order_ST", "Interaction_ST_minus_S1"]].head(6).to_string(index=False))
    print("=" * 80 + "\n")

    return df_sobol


if __name__ == "__main__":
    run_sobol_analysis(n_base_samples=2048)
