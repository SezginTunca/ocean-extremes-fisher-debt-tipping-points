#!/usr/bin/env python3
"""
compare_esm_models.py
Cross-Model Comparison of Earth System Models (GFDL-ESM4 vs. MPI-ESM1-2-HR vs. Ensemble-Mean)
for ABM-Earth-TropEcS.
Evaluates:
- Differences in warming anomaly and deoxygenation rates
- Metabolic Index (Phi) habitat compression
- Ecological tipping and fish biomass trajectories
- Socio-economic debt-trap activation rates
Exports:
- tables/Table_ESM_Model_Comparison.csv
- tables/Table_ESM_Model_Comparison.md
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add model and data to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "model"))
sys.path.append(str(PROJECT_ROOT / "data"))

from engine import ABMEarthEngine
from cmip6_downscaler import ESM_METADATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ESMComparison")

TABLES_DIR = PROJECT_ROOT / "tables"

def run_esm_comparison():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    models = ["GFDL-ESM4", "MPI-ESM1-2-HR", "Ensemble-Mean"]
    scenarios = ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    
    records = []
    
    logger.info("Executing cross-ESM simulation comparison across models and scenarios...")
    
    for model_name in models:
        meta = ESM_METADATA[model_name]
        for scen in scenarios:
            logger.info(f"Running ABM simulation: Model={model_name}, Scenario={scen} (2000-2050)...")
            engine = ABMEarthEngine(
                scenario=scen,
                esm_model=model_name,
                start_year=2000,
                end_year=2050,
                seed=42
            )
            df = engine.run_simulation()
            
            # Post-2025 projection metrics
            df_proj = df[df["year"] >= 2025]
            
            mean_sst = float(df_proj["sst_celsius"].mean())
            warming_anomaly_2050 = float(
                df[df["year"] == 2050]["sst_celsius"].mean() - df[df["year"] == 2000]["sst_celsius"].mean()
            )
            mean_phi = float(df_proj["metabolic_index_phi"].mean())
            mhw_months_pct = float(df_proj["is_mhw"].mean() * 100.0)
            final_biomass = float(df["total_biomass"].iloc[-1])
            peak_debt_trap = float(df["debt_trap_fraction"].max())
            total_blue_carbon = float(df["cumulative_blue_carbon"].iloc[-1])
            total_export_rev_musd = float(df["export_revenue_usd"].sum() / 1e6)
            
            records.append({
                "ESM_Model": model_name,
                "Ocean_BGC_Architecture": f"{meta['ocean_model']} | {meta['bgc_model'].split('(')[0].strip()}",
                "Scenario": scen,
                "Mean_SST_Celsius": round(mean_sst, 2),
                "Warming_Anomaly_2050_C": round(warming_anomaly_2050, 2),
                "Mean_Metabolic_Index_Phi": round(mean_phi, 2),
                "MHW_Frequency_Pct": round(mhw_months_pct, 1),
                "Final_Biomass_Tonnes": round(final_biomass, 1),
                "Peak_Debt_Trap_Fraction": round(peak_debt_trap, 3),
                "Total_Blue_Carbon_tCO2e": round(total_blue_carbon, 1),
                "Total_Export_Revenue_MUSD": round(total_export_rev_musd, 2)
            })
            
    df_comp = pd.DataFrame(records)
    
    # Save outputs
    csv_out = TABLES_DIR / "Table_ESM_Model_Comparison.csv"
    md_out = TABLES_DIR / "Table_ESM_Model_Comparison.md"
    
    df_comp.to_csv(csv_out, index=False)
    
    with open(md_out, "w") as f:
        f.write("# Table S4 | Cross-Model ESM Comparison: GFDL-ESM4 vs. MPI-ESM1-2-HR vs. Ensemble-Mean\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n\n*Simulated under ABM-Earth-TropEcS monthly coupled model (2000–2050).*\n")
        
    logger.info(f"Comparison saved to {csv_out} and {md_out}")
    
    # Print Executive Summary
    print("\n" + "="*85)
    print("CROSS-MODEL ESM COMPARISON (SSP2-4.5 Benchmark)")
    print("="*85)
    ssp2 = df_comp[df_comp["Scenario"] == "SSP2-4.5"]
    print(ssp2[[
        "ESM_Model", "Warming_Anomaly_2050_C", "Mean_Metabolic_Index_Phi",
        "MHW_Frequency_Pct", "Final_Biomass_Tonnes", "Peak_Debt_Trap_Fraction"
    ]].to_string(index=False))
    print("="*85 + "\n")
    
    return df_comp

if __name__ == "__main__":
    run_esm_comparison()
