#!/usr/bin/env python3
"""
analysis/mechanism_ablation.py
================================
Causal Mechanism Ablation for ABM-Earth-TropEcS under SSP5-8.5 (2000-2050).
Evaluates M = 30 Monte Carlo realizations per experimental arm to isolate the marginal
contribution of each coupled biophysical-human feedback mechanism:
1. Full Coupled ABM (Baseline SSP5-8.5)
2. No Informal Debt Lock-In (enable_debt_trap=False)
3. No Metabolic & Hypoxic Compression (enable_metabolic_compression=False)
4. No Subsidized DWF Competition (num_dwf_vessels=0, capacity_subsidy_rate=0.0)

Exports:
- tables/Table_Mechanism_Ablation.csv
- tables/Table_Mechanism_Ablation.md
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
TABLES_DIR = PROJECT_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from engine import ABMEarthEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MechanismAblation")

ABLATION_ARMS: List[Dict[str, object]] = [
    {"key": "full_coupled", "label": "Full Coupled ABM (SSP5-8.5)", "kwargs": {}},
    {"key": "no_debt_trap", "label": "No Informal Debt Lock-In", "kwargs": {"enable_debt_trap": False}},
    {"key": "no_metabolic", "label": "No Metabolic / Hypoxic Compression", "kwargs": {"enable_metabolic_compression": False}},
    {"key": "no_dwf", "label": "No Subsidized DWF Competition", "kwargs": {"num_dwf_vessels": 0, "capacity_subsidy_rate": 0.0}},
]


def run_mechanism_ablation(n_runs: int = 30, base_seed: int = 100) -> pd.DataFrame:
    """Executes M = 30 CRN stochastic simulations per ablation arm under SSP5-8.5."""
    logger.info(f"Running 4-arm SSP5-8.5 causal mechanism ablation ({n_runs} seeds per arm)...")
    arm_samples: Dict[str, Dict[str, np.ndarray]] = {}

    for arm in ABLATION_ARMS:
        bio_list, dt_list, haz_list, carb_list = [], [], [], []
        for i in range(n_runs):
            eng = ABMEarthEngine(
                scenario="SSP5-8.5",
                start_year=2000,
                end_year=2050,
                seed=base_seed + i,
                **arm["kwargs"]
            )
            df = eng.run_simulation()
            bio_list.append(float(df["total_biomass"].iloc[-1]))
            dt_list.append(float(df["debt_trap_fraction"].max()))
            haz_vals = df.loc[df["mean_hazard_trapped"] > 0, "mean_hazard_trapped"]
            haz_list.append(float(haz_vals.mean()) if len(haz_vals) > 0 else float(df["mean_hazard_solvent"].mean()))
            carb_list.append(float(df["cumulative_blue_carbon"].iloc[-1]))

        arm_samples[arm["key"]] = {
            "bio": np.array(bio_list, dtype=float),
            "dt": np.array(dt_list, dtype=float),
            "haz": np.array(haz_list, dtype=float),
            "carb": np.array(carb_list, dtype=float),
        }
        logger.info(
            f"  [{arm['label']}] Final Biomass = {np.mean(bio_list):.2f} ± {np.std(bio_list, ddof=1):.2f} t | "
            f"Peak Debt-Trap = {np.mean(dt_list):.3f}"
        )

    base_bio = arm_samples["full_coupled"]["bio"]
    base_mean = float(np.mean(base_bio))
    rows = []

    for arm in ABLATION_ARMS:
        s = arm_samples[arm["key"]]
        b_mean = float(np.mean(s["bio"]))
        b_sd = float(np.std(s["bio"], ddof=1))
        b_ci = float(1.96 * b_sd / np.sqrt(n_runs))
        dt_mean = float(np.mean(s["dt"]))
        dt_sd = float(np.std(s["dt"], ddof=1))
        haz_mean = float(np.mean(s["haz"]))
        carb_mean = float(np.mean(s["carb"]))
        delta_pct = ((b_mean - base_mean) / base_mean) * 100.0 if arm["key"] != "full_coupled" else 0.0

        if arm["key"] == "full_coupled":
            pval = np.nan
            t_stat = np.nan
        else:
            t_res = stats.ttest_rel(s["bio"], base_bio)
            t_stat = float(t_res.statistic)
            pval = float(t_res.pvalue)

        rows.append({
            "Arm_Key": arm["key"],
            "Ablation_Arm": arm["label"],
            "N_Runs": n_runs,
            "Final_Biomass_Mean_t": round(b_mean, 2),
            "Final_Biomass_SD_t": round(b_sd, 2),
            "Final_Biomass_CI95_t": round(b_ci, 2),
            "Biomass_Change_vs_Full_Pct": round(delta_pct, 2),
            "Peak_Debt_Trap_Mean": round(dt_mean, 4),
            "Peak_Debt_Trap_SD": round(dt_sd, 4),
            "Mean_Offshore_Hazard": round(haz_mean, 3),
            "Cumulative_Blue_Carbon_tCO2e": round(carb_mean, 2),
            "Paired_t_vs_Full": round(t_stat, 3) if not np.isnan(t_stat) else np.nan,
            "Paired_p_vs_Full": f"{pval:.2e}" if not np.isnan(pval) else "—",
        })

    df_ablation = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "Table_Mechanism_Ablation.csv"
    out_md = TABLES_DIR / "Table_Mechanism_Ablation.md"
    df_ablation.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("**Extended Data Table 4 | Causal Mechanism Ablation Under Unmitigated Warming (SSP5-8.5, 2000–2050, M = 30 Seeds per Arm)**\n\n")
        f.write(df_ablation.to_markdown(index=False))
        f.write("\n\n*Notes: Each arm executes M = 30 Common Random Numbers (CRN) realizations from 2000 to 2050 under GFDL-ESM4 SSP5-8.5. Two-sided paired t-tests compare terminal 2050 biomass against the Full Coupled ABM baseline.*\n")

    logger.info(f"Saved causal mechanism ablation table to {out_csv} and {out_md}")
    print("\n" + "=" * 88)
    print("SSP5-8.5 CAUSAL MECHANISM ABLATION (M = 30 CRN SEEDS PER ARM)")
    print("=" * 88)
    print(df_ablation.to_string(index=False))
    print("=" * 88 + "\n")
    return df_ablation


if __name__ == "__main__":
    run_mechanism_ablation()
