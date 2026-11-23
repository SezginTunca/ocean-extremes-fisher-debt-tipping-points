#!/usr/bin/env python3
"""
model/stochastic_evaluator.py
==============================
Multi-Run Stochastic Monte Carlo Evaluation & Confidence Envelopes for ABM-Earth-TropEcS.
Executes an ensemble of M stochastic iterations per policy scenario across randomized seeds
to quantify behavioral and market stochasticity.

Scenarios Evaluated:
1. Status Quo (Harmful subsidies active, 10% MPA, low carbon price, no safety nets)
2. WTO Subsidy Prohibition (Prohibition of capacity-enhancing subsidies, sigma = 0)
3. 30x30 Spatial Closures (30% MPA network without financial compensation)
4. Blue Carbon Dividend (Mangrove carbon crediting at $35/tCO2e + emergency social transfers)
5. Integrated Policy Package (WTO ban + 30x30 MPA + Blue Carbon Dividend)

Outputs:
- tables/Table_Stochastic_Monte_Carlo_Ensemble.csv
- tables/Table_Stochastic_Monte_Carlo_Ensemble.md
- tables/stochastic_trajectories.parquet (Time-resolved 95% confidence bands for plotting)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

try:
    from model.engine import ABMEarthEngine
except ImportError:
    from engine import ABMEarthEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StochasticEvaluator")

SCENARIOS = {
    "Status Quo": {
        "wto_prohibition": False,
        "capacity_subsidy_rate": 0.35,
        "mpa_fraction": 0.10,
        "carbon_price": 15.0,
        "safety_net_active": False
    },
    "WTO Subsidy Ban": {
        "wto_prohibition": True,
        "capacity_subsidy_rate": 0.00,
        "mpa_fraction": 0.10,
        "carbon_price": 15.0,
        "safety_net_active": False
    },
    "30x30 Spatial Closures": {
        "wto_prohibition": False,
        "capacity_subsidy_rate": 0.35,
        "mpa_fraction": 0.30,
        "carbon_price": 15.0,
        "safety_net_active": False
    },
    "Blue Carbon Dividend": {
        "wto_prohibition": False,
        "capacity_subsidy_rate": 0.35,
        "mpa_fraction": 0.10,
        "carbon_price": 35.0,
        "safety_net_active": True
    },
    "Integrated Policy Package": {
        "wto_prohibition": True,
        "capacity_subsidy_rate": 0.00,
        "mpa_fraction": 0.30,
        "carbon_price": 35.0,
        "safety_net_active": True
    }
}

def run_stochastic_ensemble(n_runs_per_scenario: int = 30, duration_years: int = 30):
    """Executes stochastic ensemble across scenarios."""
    summary_records = []
    trajectory_records = []
    
    logger.info(f"Starting stochastic evaluation: {len(SCENARIOS)} scenarios x {n_runs_per_scenario} runs...")
    
    for sc_name, sc_params in SCENARIOS.items():
        logger.info(f"Simulating scenario: [{sc_name}] across {n_runs_per_scenario} random seeds...")
        
        final_biomasses = []
        peak_debt_traps = []
        tot_carbons = []
        tot_revenues = []
        
        # Matrix to collect monthly trajectories for confidence intervals: shape (n_runs, n_months)
        bio_trajs = []
        dt_trajs = []
        
        for run_idx in range(n_runs_per_scenario):
            seed = 1000 + run_idx * 17
            engine = ABMEarthEngine(
                scenario="SSP2-4.5",
                start_year=2020,
                end_year=2020 + duration_years,
                wto_prohibition=sc_params["wto_prohibition"],
                capacity_subsidy_rate=sc_params["capacity_subsidy_rate"],
                mpa_fraction=sc_params["mpa_fraction"],
                carbon_price=sc_params["carbon_price"],
                safety_net_active=sc_params["safety_net_active"],
                seed=seed
            )
            df_sim = engine.run_simulation()
            
            final_bio = df_sim["total_biomass"].iloc[-1]
            peak_dt = df_sim["debt_trap_fraction"].max()
            tot_c = df_sim["cumulative_blue_carbon"].iloc[-1]
            tot_rev = df_sim["export_revenue_usd"].sum()
            
            final_biomasses.append(final_bio)
            peak_debt_traps.append(peak_dt)
            tot_carbons.append(tot_c)
            tot_revenues.append(tot_rev)
            
            bio_trajs.append(df_sim["total_biomass"].values)
            dt_trajs.append(df_sim["debt_trap_fraction"].values)
            
        # Compute trajectory quantiles
        bio_mat = np.array(bio_trajs)
        dt_mat = np.array(dt_trajs)
        n_months = bio_mat.shape[1]
        
        for m in range(n_months):
            trajectory_records.append({
                "Scenario": sc_name,
                "Month": m,
                "Biomass_Mean": float(np.mean(bio_mat[:, m])),
                "Biomass_Median": float(np.median(bio_mat[:, m])),
                "Biomass_Q25": float(np.percentile(bio_mat[:, m], 25)),
                "Biomass_Q75": float(np.percentile(bio_mat[:, m], 75)),
                "Biomass_CI_Low": float(np.percentile(bio_mat[:, m], 2.5)),
                "Biomass_CI_High": float(np.percentile(bio_mat[:, m], 97.5)),
                "DebtTrap_Mean": float(np.mean(dt_mat[:, m])),
                "DebtTrap_CI_Low": float(np.percentile(dt_mat[:, m], 2.5)),
                "DebtTrap_CI_High": float(np.percentile(dt_mat[:, m], 97.5))
            })
            
        summary_records.append({
            "Scenario": sc_name,
            "Runs": n_runs_per_scenario,
            "Final_Biomass_Mean": round(float(np.mean(final_biomasses)), 2),
            "Final_Biomass_SD": round(float(np.std(final_biomasses)), 2),
            "Final_Biomass_95CI": f"[{np.percentile(final_biomasses, 2.5):.1f}, {np.percentile(final_biomasses, 97.5):.1f}]",
            "Peak_Debt_Trap_Mean": round(float(np.mean(peak_debt_traps)), 3),
            "Peak_Debt_Trap_SD": round(float(np.std(peak_debt_traps)), 3),
            "Peak_Debt_Trap_95CI": f"[{np.percentile(peak_debt_traps, 2.5):.3f}, {np.percentile(peak_debt_traps, 97.5):.3f}]",
            "Cumulative_Carbon_Mean": round(float(np.mean(tot_carbons)), 1),
            "Export_Revenue_MUSD": round(float(np.mean(tot_revenues) / 1e6), 2)
        })
        
    df_summary = pd.DataFrame(summary_records)
    df_trajs = pd.DataFrame(trajectory_records)
    
    out_csv = TABLES_DIR / "Table_Stochastic_Monte_Carlo_Ensemble.csv"
    out_md = TABLES_DIR / "Table_Stochastic_Monte_Carlo_Ensemble.md"
    out_parquet = TABLES_DIR / "stochastic_trajectories.parquet"
    
    df_summary.to_csv(out_csv, index=False)
    df_trajs.to_parquet(out_parquet, index=False)
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("**Table S5 | Multi-Run Stochastic Monte Carlo Evaluation Across Policy Packages (M = 30 Iterations per Scenario)**\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n*Notes: Means, standard deviations (SD), and empirical 95% confidence intervals [Q2.5, Q97.5] estimated across M = 30 stochastic runs per policy package with randomized micro-economic shocks and agent behavioral seeds. Simulates 25 years (300 months) under SSP2-4.5 climate forcing.*")
        
    logger.info(f"Saved stochastic ensemble summary to {out_csv} and {out_md}")
    logger.info(f"Saved time-resolved stochastic trajectories to {out_parquet}")
    
    print("\n" + "="*80)
    print("STOCHASTIC MONTE CARLO ENSEMBLE SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")
    
    return df_summary, df_trajs

if __name__ == "__main__":
    run_stochastic_ensemble(n_runs_per_scenario=30, duration_years=25)
