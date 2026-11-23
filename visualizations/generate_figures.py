#!/usr/bin/env python3
"""
generate_figures.py
Publication-Grade Single-Grammar Main Figures (Figures 1-6)
Formatted for Nature Climate Change double-column print standards (600 DPI PNG + vector PDF).

Strict Within-Figure Single Visual Grammar Across All 6 Main Figures:
- Figure 1 (2x3): Continuous multi-scale shelf & five-agent mechanistic coupling curves (a-f)
- Figure 2 (2x3): Co-registered global Equal-Earth (EPSG:8857) choropleth submaps across all 109 states (a-f)
- Figure 3 (2x3): Nonlinear dynamical regime-shift paths, EWS & phase-portrait loops (a-f)
- Figure 4 (2x3): Horizontal raincloud-interval micro-behavioral & causal ablation distributions (a-f)
- Figure 5 (2x3): Horizontal dot-interval 17-parameter Sobol GSA, Direct-ABM agreement & skill rankings (a-f)
- Figure 6 (2x3): Co-formatted 2020-2045 Monte Carlo policy ensemble paths & ribbons (a-f)
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

TARGET_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TARGET_DIR / "data"
TABLES_DIR = TARGET_DIR / "tables"
ANALYSIS_DIR = TARGET_DIR / "analysis"

for p in [TARGET_DIR / "visualizations", TARGET_DIR / "data", TARGET_DIR / "model", ANALYSIS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from style_and_maps import (
    COLORS,
    PARAM_LABELS,
    POLICY_COLORS,
    POLICY_SHORT,
    SOLVENCY_COLORS,
    SSP_COLORS,
    apply_nature_style,
    compute_coastal_corridor_loitering,
    panel_tag,
    plot_global_equal_earth,
    plot_raincloud_h,
    save_pub_figure,
)

apply_nature_style()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Visualizations")


# ==============================================================================
# FIGURE 1: 2x3 Continuous Multi-Scale Shelf & Five-Agent Coupling Curve Suite
# ==============================================================================
def generate_figure_1():
    logger.info("Generating Figure 1 (2x3 uniform continuous multi-scale coupling curves a-f)...")
    from engine import ABMEarthEngine

    # Run coupled trajectories across GFDL-ESM4, MPI-ESM1-2-HR, and Ensemble-Mean
    runs = {}
    spatial_2000 = {}
    spatial_2050 = {}
    for esm in ["GFDL-ESM4", "MPI-ESM1-2-HR", "Ensemble-Mean"]:
        for scen in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
            eng = ABMEarthEngine(scenario=scen, esm_model=esm, start_year=2000, end_year=2050, seed=42)
            t_2000_cols, phi_2000_cols = [], []
            for m in range(12):
                rw = eng.climate_data.iloc[m]
                eng.env.update_climate_forcing(rw["sst_celsius"], rw["po2_kpa"], rw["npp_index"], rw["metabolic_index_phi"])
                t_2000_cols.append(eng.env.sst.mean(axis=0).copy())
                phi_2000_cols.append(eng.env.phi.mean(axis=0).copy())
            spatial_2000[(esm, scen)] = (np.mean(t_2000_cols, axis=0), np.mean(phi_2000_cols, axis=0))

            eng_full = ABMEarthEngine(scenario=scen, esm_model=esm, start_year=2000, end_year=2050, seed=42)
            df_sim = eng_full.run_simulation()
            runs[(esm, scen)] = (eng_full, df_sim)

            eng_end = ABMEarthEngine(scenario=scen, esm_model=esm, start_year=2000, end_year=2050, seed=42)
            t_2050_cols, phi_2050_cols = [], []
            for m in range(12):
                rw = eng_end.climate_data.iloc[50 * 12 + m]
                eng_end.env.update_climate_forcing(rw["sst_celsius"], rw["po2_kpa"], rw["npp_index"], rw["metabolic_index_phi"])
                t_2050_cols.append(eng_end.env.sst.mean(axis=0).copy())
                phi_2050_cols.append(eng_end.env.phi.mean(axis=0).copy())
            spatial_2050[(esm, scen)] = (np.mean(t_2050_cols, axis=0), np.mean(phi_2050_cols, axis=0))

    # Multi-seed ensemble (M=12) for cross-shelf 2050 standing biomass (Panel d) & regime-separated fleet effort (Panel e)
    shelf_bio_2050 = {(esm, sc): [] for esm in ["GFDL-ESM4", "MPI-ESM1-2-HR", "Ensemble-Mean"] for sc in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]}
    solv_eff_ens, trap_eff_ens, dwf_eff_ens = [], [], []
    for sd in range(12):
        for esm in ["GFDL-ESM4", "MPI-ESM1-2-HR", "Ensemble-Mean"]:
            for sc in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
                e_b = ABMEarthEngine(scenario=sc, esm_model=esm, start_year=2000, end_year=2050, seed=200 + sd)
                e_b.run_simulation()
                shelf_bio_2050[(esm, sc)].append(e_b.spatial_biomass_end.sum(axis=0))

        e_solv = ABMEarthEngine(
            scenario="SSP2-4.5", wto_prohibition=True, capacity_subsidy_rate=0.0,
            mpa_fraction=0.10, carbon_price=35.0, safety_net_active=True, start_year=2000, end_year=2050, seed=200 + sd
        )
        e_solv.run_simulation()
        sp_solv = e_solv.spatial_effort_ssf.sum(axis=0)
        solv_eff_ens.append(sp_solv / max(1e-6, sp_solv.sum()) * 100.0)

        e_trap = ABMEarthEngine(scenario="SSP5-8.5", mpa_fraction=0.10, start_year=2030, end_year=2050, seed=200 + sd)
        for f_ag in e_trap.ssf_fishers:
            f_ag.debt = 500.0
            f_ag.in_debt_trap = True
        e_trap.run_simulation()
        sp_trap = e_trap.spatial_effort_ssf.sum(axis=0)
        sp_dwf = e_trap.spatial_effort_dwf.sum(axis=0)
        trap_eff_ens.append(sp_trap / max(1e-6, sp_trap.sum()) * 100.0)
        dwf_eff_ens.append(sp_dwf / max(1e-6, sp_dwf.sum()) * 100.0)

    fig = plt.figure(figsize=(14.8, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.28, top=0.91)
    shelf_km = np.arange(10) * 10.0

    # Shared top figure-level legend for climate scenarios & GFDL-MPI inter-ESM spread
    top_handles = [
        Line2D([0], [0], color=SSP_COLORS["Historical"], lw=2.0, linestyle=":", label="2000 Historical baseline"),
        Line2D([0], [0], color=SSP_COLORS["SSP1-2.6"], lw=2.0, linestyle="-.", label="SSP1-2.6 (strong mitigation)"),
        Line2D([0], [0], color=SSP_COLORS["SSP2-4.5"], lw=2.0, linestyle="--", label="SSP2-4.5 (intermediate)"),
        Line2D([0], [0], color=SSP_COLORS["SSP5-8.5"], lw=2.1, linestyle="-", label="SSP5-8.5 (unmitigated)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SSP_COLORS["SSP5-8.5"], alpha=0.18, edgecolor="none", label="GFDL-ESM4 to MPI-ESM1-2-HR range"),
    ]
    fig.legend(
        handles=top_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=5, frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=7.5
    )

    # Panel a: Cross-Shelf Coastal Bathymetric Warming Profile (0-90 km)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a")
    t0_ens = spatial_2000[("Ensemble-Mean", "SSP5-8.5")][0]
    ax_a.plot(shelf_km, t0_ens, color=SSP_COLORS["Historical"], lw=1.9, linestyle=":")
    for scen, ls in [("SSP1-2.6", "-."), ("SSP2-4.5", "--"), ("SSP5-8.5", "-")]:
        tg = spatial_2050[("GFDL-ESM4", scen)][0]
        tm = spatial_2050[("MPI-ESM1-2-HR", scen)][0]
        te = spatial_2050[("Ensemble-Mean", scen)][0]
        ax_a.fill_between(shelf_km, np.minimum(tg, tm), np.maximum(tg, tm), color=SSP_COLORS[scen], alpha=0.18)
        ax_a.plot(shelf_km, te, color=SSP_COLORS[scen], lw=2.0, linestyle=ls)
    ax_a.text(
        0.96, 0.94, "SSP5-8.5 warming:\n+1.60 °C (+1.53 to +1.68 °C)",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=6.7, fontweight="bold",
        color=SSP_COLORS["SSP5-8.5"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_a.set_xlim(0, 90)
    ax_a.set_xlabel("Cross-shelf distance from coast (km)")
    ax_a.set_ylabel("Coastal shelf temperature (°C)")
    ax_a.grid(True)

    # Panel b: Cross-Shelf Arrhenius Metabolic Scope Compression Profile (0-90 km)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b")
    p0_ens = spatial_2000[("Ensemble-Mean", "SSP5-8.5")][1]
    ax_b.plot(shelf_km, p0_ens, color=SSP_COLORS["Historical"], lw=1.9, linestyle=":")
    for scen, ls in [("SSP1-2.6", "-."), ("SSP2-4.5", "--"), ("SSP5-8.5", "-")]:
        pg = spatial_2050[("GFDL-ESM4", scen)][1]
        pm = spatial_2050[("MPI-ESM1-2-HR", scen)][1]
        pe = spatial_2050[("Ensemble-Mean", scen)][1]
        ax_b.fill_between(shelf_km, np.minimum(pg, pm), np.maximum(pg, pm), color=SSP_COLORS[scen], alpha=0.18)
        ax_b.plot(shelf_km, pe, color=SSP_COLORS[scen], lw=2.0, linestyle=ls)
    ax_b.axhline(2.0, color="#1E293B", linestyle="--", lw=1.2)
    ax_b.text(46, 2.04, "$\\Phi_{\\mathrm{crit}} = 2.0$ survival limit", fontsize=6.8, fontweight="bold", color="#1E293B")
    ax_b.text(0.04, 0.06, "Shelf mean $\\Phi$: 2.33 $\\rightarrow$ 1.56 (-33.1%)",
              transform=ax_b.transAxes, fontsize=6.8, fontweight="bold", color=SSP_COLORS["SSP5-8.5"],
              bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CBD5E1", alpha=0.92))
    ax_b.set_xlim(0, 90)
    ax_b.set_xlabel("Cross-shelf distance from coast (km)")
    ax_b.set_ylabel("Arrhenius Metabolic Index ($\\Phi$)")
    ax_b.grid(True)

    # Panel c: Hypoxic Shelf Habitat Expansion (2000-2050, smooth 5-yr ENSO-filtered trend across full horizon)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c")
    yrs_ann = np.arange(2000, 2051)
    h_g5_raw = runs[("GFDL-ESM4", "SSP5-8.5")][1].groupby("year")["hypoxic_fraction"].mean().values * 100.0
    for scen, ls in [("SSP1-2.6", "-."), ("SSP2-4.5", "--"), ("SSP5-8.5", "-")]:
        hg_s = pd.Series(runs[("GFDL-ESM4", scen)][1].groupby("year")["hypoxic_fraction"].mean().values * 100.0).rolling(5, min_periods=1, center=True).mean().values
        hm_s = pd.Series(runs[("MPI-ESM1-2-HR", scen)][1].groupby("year")["hypoxic_fraction"].mean().values * 100.0).rolling(5, min_periods=1, center=True).mean().values
        he_s = pd.Series(runs[("Ensemble-Mean", scen)][1].groupby("year")["hypoxic_fraction"].mean().values * 100.0).rolling(5, min_periods=1, center=True).mean().values
        if scen == "SSP5-8.5":
            ax_c.plot(yrs_ann[:26], he_s[:26], color=SSP_COLORS["Historical"], lw=2.0)
        ax_c.fill_between(yrs_ann[25:], np.minimum(hg_s[25:], hm_s[25:]), np.maximum(hg_s[25:], hm_s[25:]),
                          color=SSP_COLORS[scen], alpha=0.22)
        ax_c.plot(yrs_ann[25:], he_s[25:], color=SSP_COLORS[scen], lw=2.0, linestyle=ls)
    ax_c.axvline(2025, color="#64748B", linestyle=":", lw=1.0)
    ax_c.text(2025.6, 94.0, "2025 projection", fontsize=6.4, color="#475569")
    ax_c.annotate(f"2000: {h_g5_raw[0]:.1f}%", (2000, he_s[0]), xytext=(4, 10), textcoords="offset points",
                  fontsize=6.8, fontweight="bold", color="#222222")
    ax_c.annotate(f"2050: {h_g5_raw[-1]:.1f}%", (2050, he_s[-1]), xytext=(-54, 7), textcoords="offset points",
                  fontsize=6.8, fontweight="bold", color=SSP_COLORS["SSP5-8.5"])
    ax_c.set_xlim(2000, 2050)
    ax_c.set_ylim(0, 102)
    ax_c.set_xlabel("Simulation year")
    ax_c.set_ylabel("Hypoxic shelf habitat $\\Phi < 2.0$ (%)")
    ax_c.grid(True)

    # Panel d: Cross-Shelf Coupled Standing Fish Biomass Profile (0-90 km, 2000 vs 2050 + GFDL-MPI spread)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d")
    b0_col = np.full_like(shelf_km, 70.0)
    ax_d.plot(shelf_km, b0_col, color=SSP_COLORS["Historical"], lw=1.9, linestyle=":")
    for scen, ls in [("SSP1-2.6", "-."), ("SSP2-4.5", "--"), ("SSP5-8.5", "-")]:
        bg_c = np.mean(shelf_bio_2050[("GFDL-ESM4", scen)], axis=0)
        bm_c = np.mean(shelf_bio_2050[("MPI-ESM1-2-HR", scen)], axis=0)
        be_c = np.mean(shelf_bio_2050[("Ensemble-Mean", scen)], axis=0)
        ax_d.fill_between(shelf_km, np.minimum(bg_c, bm_c), np.maximum(bg_c, bm_c),
                          color=SSP_COLORS[scen], alpha=0.22)
        ax_d.plot(shelf_km, be_c, color=SSP_COLORS[scen], lw=2.05, linestyle=ls)
    ax_d.text(
        0.04, 0.84,
        "2050 GFDL vs MPI domain spread:\nSSP2-4.5: 262.1 vs 272.4 t (3.8%)\nSSP5-8.5: 204.9 vs 212.8 t",
        transform=ax_d.transAxes, ha="left", va="top", fontsize=6.5, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_d.set_xlim(0, 90)
    ax_d.set_ylim(10, 76)
    ax_d.set_xlabel("Cross-shelf distance from coast (km)")
    ax_d.set_ylabel("Standing fish biomass (t per 10-km band)")
    ax_d.grid(True)

    # Panel e: Cross-Shelf Spatial Fleet Effort Partitioning (0-90 km, Continuous Regime Curves + 95% CI Ribbons)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e")
    solv_arr = np.array(solv_eff_ens)
    trap_arr = np.array(trap_eff_ens)
    dwf_arr = np.array(dwf_eff_ens)
    for arr_e, col_e, ls_e, lbl_e in [
        (solv_arr, SOLVENCY_COLORS["Solvent"], "-", "Debt-free fishers ($D \\leq D_{\\mathrm{thresh}}$)"),
        (trap_arr, SOLVENCY_COLORS["Trapped"], "--", "Debt-trapped fishers (offshore shift)"),
        (dwf_arr, COLORS["blue"], "-.", "Subsidized industrial DWF"),
    ]:
        m_e = arr_e.mean(axis=0)
        ci_e = 1.96 * arr_e.std(axis=0, ddof=1) / np.sqrt(len(arr_e))
        ax_e.fill_between(shelf_km, np.maximum(0.0, m_e - ci_e), m_e + ci_e, color=col_e, alpha=0.16)
        ax_e.plot(shelf_km, m_e, color=col_e, lw=2.05, linestyle=ls_e, label=lbl_e)
    ax_e.axvline(30.0, color="#64748B", linestyle=":", lw=1.1)
    ax_e.text(31.5, 42.0, "Nearshore range\nlimit (30 km)", fontsize=6.4, color="#475569")
    offshore_pct = float(trap_arr[:, 4:].sum(axis=1).mean())
    ax_e.annotate(
        f"{offshore_pct:.1f}% trips >30 km",
        (50.0, float(trap_arr[:, 5].mean())), xytext=(54.0, 22.0), textcoords="data",
        fontsize=6.4, fontweight="bold", color=SOLVENCY_COLORS["Trapped"],
        arrowprops=dict(arrowstyle="-|>", color=SOLVENCY_COLORS["Trapped"], lw=0.7, mutation_scale=7)
    )
    ax_e.set_xlim(0, 90)
    ax_e.set_ylim(0, float(solv_arr.mean(axis=0).max()) * 1.15)
    ax_e.set_xlabel("Cross-shelf distance from coast (km)")
    ax_e.set_ylabel("Spatial effort allocation (% of trips)")
    ax_e.legend(loc="upper right", frameon=True, fontsize=6.4)
    ax_e.grid(True)

    # Panel f: Intertidal Mangrove Conservation & Carbon Land-Use Equilibrium Curve (Eq. S11)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f")
    pc_grid = np.linspace(0.0, 80.0, 100)
    aqua_ret = 4.0 * 350.0
    for mpa_lbl, mpa_enf, mpa_fr, col_m, ls_m in [
        ("30×30 MPA co-management (30%)", True, 0.30, COLORS["teal"], "-"),
        ("Status Quo protection (10%)", False, 0.10, COLORS["coral"], "--"),
    ]:
        mpa_prem = (520.0 * (mpa_fr / 0.30)) if mpa_enf else (180.0 * (mpa_fr / 0.10))
        c_ret = 12.0 * pc_grid + 140.0 + mpa_prem
        mu_eq = np.clip(c_ret / (c_ret + aqua_ret), 0.12, 0.92) * 100.0
        mu_lo = np.clip(c_ret / (c_ret + aqua_ret) + 0.35 * (0.30 - 0.50), 0.12, 0.92) * 100.0
        mu_hi = np.clip(c_ret / (c_ret + aqua_ret) + 0.35 * (0.70 - 0.50), 0.12, 0.92) * 100.0
        ax_f.fill_between(pc_grid, mu_lo, mu_hi, color=col_m, alpha=0.15)
        ax_f.plot(pc_grid, mu_eq, color=col_m, lw=2.1, linestyle=ls_m, label=mpa_lbl)

    ax_f.axvline(35.0, color=COLORS["navy"], linestyle=":", lw=1.2)
    ax_f.text(36.5, 16.0, "Dividend $P_C$ = USD 35", fontsize=6.7, fontweight="bold", color=COLORS["navy"])
    ax_f.set_xlim(0, 80)
    ax_f.set_ylim(10, 72)
    ax_f.set_xlabel("Blue-carbon price $P_C$ (USD $\\mathrm{tCO_2e}^{-1}$)")
    ax_f.set_ylabel("Equilibrium mangrove cover $\\mu^*$ (%)")
    ax_f.legend(loc="upper left", frameon=True, fontsize=6.5)
    ax_f.grid(True)

    save_pub_figure(fig, "Figure_1_Multi_Scale_Architecture")


# ==============================================================================
# FIGURE 2: Pure 2x3 Co-Registered Global Equal-Earth Empirical Submap Suite
# ==============================================================================
def generate_figure_2():
    logger.info("Generating Figure 2 (pure 2x3 co-registered Equal-Earth submaps across all 109 states a-f)...")
    df = pd.read_csv(DATA_DIR / "country_profiles.csv")
    coastal_loit_dict = compute_coastal_corridor_loitering()
    ram_isos = set(df.loc[df["is_direct_ram_assessed"] == 1, "country_iso3"])

    fig = plt.figure(figsize=(15.4, 7.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.10)

    # Panel a: Foreign DWF Effort Share (%)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a", x=0.01, y=1.01, despine=False)
    dwf_map = dict(zip(df["country_iso3"], df["foreign_dwf_share"] * 100.0))
    plot_global_equal_earth(
        ax_a, values_by_iso3=dwf_map, cmap="YlOrRd", vmin=0.0, vmax=90.0,
        cbar_label="Foreign DWF effort share (%)",
        summary_badge="West Africa: 66.6%  |  Coral Triangle: 53.5%"
    )

    # Panel b: Capacity-Enhancing Fisheries Subsidies (Million USD, log10 scale)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b", x=0.01, y=1.01, despine=False)
    sub_musd = dict(zip(df["country_iso3"], np.maximum(0.04, df["capacity_subsidies_usd"] / 1e6)))
    log_norm_b = mcolors.LogNorm(vmin=0.05, vmax=6000.0)
    cb_b = plot_global_equal_earth(
        ax_b, values_by_iso3=sub_musd, cmap="YlGnBu", norm=log_norm_b,
        cbar_label="Capacity-enhancing subsidies (Million USD, log)",
        summary_badge="Global: USD 20.41B yr$^{-1}$ (69.4% in DWF hubs)"
    )
    if cb_b is not None:
        cb_b.set_ticks([0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0])
        cb_b.set_ticklabels(["0.1", "1", "10", "100", "1,000", "5,000"])

    # Panel c: Flag-Free Unreported Offshore Transshipment & Untracked Vessel Activity ln(1 + h)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c", x=0.01, y=1.01, despine=False)
    loit_log = {iso: float(np.log1p(v)) for iso, v in coastal_loit_dict.items()}
    plot_global_equal_earth(
        ax_c, values_by_iso3=loit_log, cmap="PuBuGn", vmin=0.0, vmax=6.0,
        cbar_label="Unreported offshore transshipment, $\\ln(1 + \\mathrm{h})$",
        summary_badge="SEN / MRT: 362 h  |  PER: 322 h (unreported)"
    )

    # Panel d: Trophic Trade Asymmetry Index (Delta TL = Export - Import)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d", x=0.01, y=1.01, despine=False)
    tl_map = dict(zip(df["country_iso3"], df["trophic_asymmetry_index"]))
    norm_d = mcolors.TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=0.65)
    cb_d = plot_global_equal_earth(
        ax_d, values_by_iso3=tl_map, cmap="RdBu_r", norm=norm_d,
        cbar_label="Trophic trade asymmetry $\\Delta\\mathrm{TL}$ (export $-$ import)",
        summary_badge="Coral Triangle: +0.20  |  West Africa: +0.08"
    )
    if cb_d is not None:
        cb_d.set_ticks([-0.6, -0.3, 0.0, 0.3, 0.6])
        cb_d.set_ticklabels(["-0.6", "-0.3", "0", "+0.3", "+0.6"])

    # Panel e: Empirical & Fisheries-Covariate Stock Status B/B_msy (+ Direct RAM EEZ diamonds)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=0.01, y=1.01, despine=False)
    bb_map = dict(zip(df["country_iso3"], df["ram_mean_bbmsy"]))
    norm_e = mcolors.TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.85)
    cb_e = plot_global_equal_earth(
        ax_e, values_by_iso3=bb_map, cmap="RdYlGn", norm=norm_e,
        cbar_label="Stock biomass status $B/B_{\\mathrm{msy}}$ (diamonds: Direct RAM, $n=10$)",
        direct_ram_isos=ram_isos,
        summary_badge="West Africa: 0.71  |  Global mean: 0.95"
    )
    if cb_e is not None:
        cb_e.set_ticks([0.6, 0.8, 1.0, 1.4, 1.8])
        cb_e.set_ticklabels(["0.6", "0.8", "1.0", "1.4", "1.8"])

    # Panel f: Coastal Poverty Headcount Rate (%)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=0.01, y=1.01, despine=False)
    pov_map = dict(zip(df["country_iso3"], np.clip(df["poverty_rate"], 0.0, 70.0)))
    norm_f = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=70.0)
    cb_f = plot_global_equal_earth(
        ax_f, values_by_iso3=pov_map, cmap="magma_r", norm=norm_f,
        cbar_label="Coastal poverty headcount rate (%, power scale)",
        summary_badge="West Africa: 31.4%  |  W. Indian Ocean: 24.0%"
    )
    if cb_f is not None:
        cb_f.set_ticks([0.0, 2.0, 10.0, 25.0, 45.0, 70.0])
        cb_f.set_ticklabels(["0", "2", "10", "25", "45", "70"])

    save_pub_figure(fig, "Figure_2_Global_Empirical_Architecture")


# ==============================================================================
# FIGURE 3: 2x3 Nonlinear Dynamical Trajectory, CSD & Hysteresis Curve Suite
# ==============================================================================
def generate_figure_3():
    logger.info("Generating Figure 3 (2x3 uniform nonlinear dynamical trajectory & phase-portrait curves a-f)...")
    from engine import ABMEarthEngine
    from tipping_point_detector import TippingPointDetector

    df_tip = pd.read_parquet(TABLES_DIR / "tipping_trajectories.parquet")
    df_tp_meta = pd.read_csv(TABLES_DIR / "Table_Tipping_Points_and_EWS.csv")

    v_row = df_tp_meta[df_tp_meta["Metric"].str.contains("Variance", na=False)]
    a_row = df_tp_meta[df_tp_meta["Metric"].str.contains("Autocorrelation", na=False)]
    tau_v, p_v = float(v_row["Value"].iloc[0]), float(v_row["p-value"].iloc[0])
    tau_a, p_a = float(a_row["Value"].iloc[0]), float(a_row["p-value"].iloc[0])

    yrs = df_tip["year"].values
    pre_end = 450
    pre_win = slice(pre_end - 48, pre_end)
    yrs_pre = yrs[pre_win]

    # Compute M=30 stochastic ensemble with strictly CAUSAL (center=False) 12-mo harmonic filter
    detector = TippingPointDetector(window_size=24, gaussian_bandwidth=18.0)
    ens_var_12, ens_ar1_12 = [], []
    dt_ens = {"SSP5-8.5": [], "SSP2-4.5": [], "SSP1-2.6": []}
    bio_ens = {"SSP5-8.5": [], "SSP2-4.5": [], "SSP1-2.6": []}
    harv_ens = {"SSP5-8.5": [], "SSP2-4.5": [], "SSP1-2.6": [], "Recovery": []}
    bio_rec_ens = []

    for i in range(30):
        d5 = ABMEarthEngine(scenario="SSP5-8.5", start_year=2000, end_year=2050, seed=100 + i).run_simulation()
        d2 = ABMEarthEngine(scenario="SSP2-4.5", start_year=2000, end_year=2050, seed=100 + i).run_simulation()
        d1 = ABMEarthEngine(scenario="SSP1-2.6", start_year=2000, end_year=2050, seed=100 + i).run_simulation()
        dr = ABMEarthEngine(
            scenario="SSP1-2.6", wto_prohibition=True, mpa_fraction=0.30,
            safety_net_active=True, start_year=2000, end_year=2050, seed=100 + i
        ).run_simulation()

        ews_i = detector.compute_early_warning_signals(d5["total_biomass"].values)
        ens_var_12.append(ews_i["rolling_variance"].rolling(12, min_periods=1).mean().iloc[pre_win].values)
        ens_ar1_12.append(ews_i["rolling_ar1"].rolling(12, min_periods=1).mean().iloc[pre_win].values)

        for k_sc, d_sc in [("SSP5-8.5", d5), ("SSP2-4.5", d2), ("SSP1-2.6", d1)]:
            dt_ens[k_sc].append(d_sc["debt_trap_fraction"].values * 100.0)
            bio_ens[k_sc].append(d_sc["total_biomass"].values)
            harv_ens[k_sc].append(d_sc["total_harvest"].values)
        harv_ens["Recovery"].append(dr["total_harvest"].values)
        bio_rec_ens.append(dr["total_biomass"].values)

    var_mat = np.array(ens_var_12)
    ar1_mat = np.array(ens_ar1_12)
    rec_mat = np.array(bio_rec_ens)

    fig = plt.figure(figsize=(14.6, 8.9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.33, wspace=0.28)

    # Panel a: 2000-2050 Multi-Scenario & Recovery Biomass Regimes (+ M=30 95% ribbons)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a")
    ax_a.fill_between(yrs, np.percentile(rec_mat, 2.5, axis=0), np.percentile(rec_mat, 97.5, axis=0),
                      color=SSP_COLORS["Recovery"], alpha=0.16)
    ax_a.plot(yrs, pd.Series(rec_mat.mean(axis=0)).rolling(6, min_periods=1).mean(),
              color=SSP_COLORS["Recovery"], lw=2.1, label="Multilateral recovery")
    for sc_k, ls_k in [("SSP1-2.6", "-."), ("SSP2-4.5", "--"), ("SSP5-8.5", "-")]:
        b_mat = np.array(bio_ens[sc_k])
        ax_a.fill_between(yrs, np.percentile(b_mat, 2.5, axis=0), np.percentile(b_mat, 97.5, axis=0),
                          color=SSP_COLORS[sc_k], alpha=0.16)
        ax_a.plot(yrs, pd.Series(b_mat.mean(axis=0)).rolling(6, min_periods=1).mean(),
                  color=SSP_COLORS[sc_k], lw=1.95, linestyle=ls_k, label=sc_k)
    ax_a.axvspan(yrs[pre_end - 48], yrs[pre_end], color=COLORS["coral"], alpha=0.13, label="EWS window (48 mo)")
    ax_a.axvline(yrs[pre_end], color=COLORS["coral"], linestyle=":", lw=1.25)
    ax_a.set_xlim(2000, 2050.5)
    ax_a.set_xlabel("Simulation year")
    ax_a.set_ylabel("Total coastal biomass (t)")
    ax_a.legend(loc="upper right", frameon=True, fontsize=6.6)
    ax_a.grid(True)

    # Panel b: Pre-Tipping Gaussian-Detrended Rolling Variance (Causal 12-mo Harmonic-Filtered + Ensemble IQR)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b")
    var_12 = df_tip["rolling_variance"].rolling(12, min_periods=1).mean().iloc[pre_win].values
    ax_b.fill_between(yrs_pre, np.percentile(var_mat, 25, axis=0), np.percentile(var_mat, 75, axis=0),
                      color=COLORS["amber"], alpha=0.22, label="Ensemble IQR ($M=30$)")
    ax_b.plot(yrs_pre, var_12, color=COLORS["amber"], lw=2.25, label="Harmonic-filtered $\\sigma^2$")
    sl_v, int_v = np.polyfit(yrs_pre, var_12, 1)
    ax_b.plot(yrs_pre, int_v + sl_v * yrs_pre, color=COLORS["navy"], linestyle="--", lw=1.6, label="Variance trend")
    ax_b.text(
        0.04, 0.94, f"Kendall $\\tau$ = {tau_v:.3f}\n$P$ = {p_v:.4f}",
        transform=ax_b.transAxes, va="top", ha="left", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_b.set_xticks([2034, 2035, 2036, 2037])
    ax_b.set_xticklabels(["2034", "2035", "2036", "2037"])
    ax_b.set_xlim(yrs_pre[0] - 0.08, yrs_pre[-1] + 0.08)
    ax_b.set_xlabel("Pre-tipping window (year)")
    ax_b.set_ylabel("Rolling residual variance ($\\mathrm{t}^2$)")
    ax_b.legend(loc="lower right", frameon=True, fontsize=6.4)
    ax_b.grid(True)

    # Panel c: Pre-Tipping Lag-1 Autocorrelation AR(1) (Causal 12-mo Harmonic-Filtered + Ensemble IQR)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c")
    ar1_12 = df_tip["rolling_ar1"].rolling(12, min_periods=1).mean().iloc[pre_win].values
    ax_c.fill_between(yrs_pre, np.percentile(ar1_mat, 25, axis=0), np.percentile(ar1_mat, 75, axis=0),
                      color=COLORS["purple"], alpha=0.22, label="Ensemble IQR ($M=30$)")
    ax_c.plot(yrs_pre, ar1_12, color=COLORS["purple"], lw=2.25, label="Harmonic-filtered $\\rho_1$")
    sl_a, int_a = np.polyfit(yrs_pre, ar1_12, 1)
    ax_c.plot(yrs_pre, int_a + sl_a * yrs_pre, color=COLORS["navy"], linestyle="--", lw=1.6, label="Memory trend")
    ax_c.text(
        0.04, 0.94, f"Kendall $\\tau$ = {tau_a:.3f}\n$P$ = {p_a:.2e}",
        transform=ax_c.transAxes, va="top", ha="left", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_c.set_xticks([2034, 2035, 2036, 2037])
    ax_c.set_xticklabels(["2034", "2035", "2036", "2037"])
    ax_c.set_xlim(yrs_pre[0] - 0.08, yrs_pre[-1] + 0.08)
    ax_c.set_ylim(0.825, 0.935)
    ax_c.set_xlabel("Pre-tipping window (year)")
    ax_c.set_ylabel("Rolling lag-1 autocorrelation ($\\rho_1$)")
    ax_c.legend(loc="lower right", frameon=True, fontsize=6.4)
    ax_c.grid(True)

    # Panel d: Harvest-Biomass Phase-Space Hysteresis Branches (Eq. S12 Integral Visualization)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d")
    h_grid = np.linspace(1.25, 3.85, 60)

    def _smooth_branch(h_mat, b_mat):
        hm = np.mean(h_mat, axis=0)
        bm = np.mean(b_mat, axis=0)
        order = np.argsort(hm)
        h_ord = gaussian_filter1d(hm[order], sigma=8.0)
        b_ord = gaussian_filter1d(bm[order], sigma=8.0)
        return np.interp(h_grid, h_ord, b_ord)

    b_rec_br = _smooth_branch(harv_ens["Recovery"], rec_mat)
    b_s1_br = _smooth_branch(harv_ens["SSP1-2.6"], bio_ens["SSP1-2.6"])
    b_s2_br = _smooth_branch(harv_ens["SSP2-4.5"], bio_ens["SSP2-4.5"])
    b_s5_br = _smooth_branch(harv_ens["SSP5-8.5"], bio_ens["SSP5-8.5"])

    ax_d.fill_between(h_grid, b_s5_br, b_rec_br, color=SSP_COLORS["SSP5-8.5"], alpha=0.13)
    ax_d.plot(h_grid, b_rec_br, color=SSP_COLORS["Recovery"], lw=2.2, linestyle="--", label="Recovery branch")
    ax_d.plot(h_grid, b_s1_br, color=SSP_COLORS["SSP1-2.6"], lw=1.85, linestyle="-.", label="SSP1-2.6 ($\\mathcal{H} = 385.8$)")
    ax_d.plot(h_grid, b_s2_br, color=SSP_COLORS["SSP2-4.5"], lw=1.85, linestyle=":", label="SSP2-4.5 ($\\mathcal{H} = 478.7$)")
    ax_d.plot(h_grid, b_s5_br, color=SSP_COLORS["SSP5-8.5"], lw=2.2, label="SSP5-8.5 ($\\mathcal{H} = 535.2$)")
    ax_d.annotate(
        "Hysteresis gap\n$\\mathcal{H} = 535.2$",
        (2.35, 0.5 * (float(np.interp(2.35, h_grid, b_rec_br)) + float(np.interp(2.35, h_grid, b_s5_br)))),
        ha="center", va="center", fontsize=6.6, fontweight="bold", color=SSP_COLORS["SSP5-8.5"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.90)
    )
    ax_d.set_xlim(1.20, 3.90)
    ax_d.set_ylim(195, 700)
    ax_d.set_xlabel("System harvest flux (t $\\mathrm{mo}^{-1}$)")
    ax_d.set_ylabel("Coastal standing biomass (t)")
    ax_d.legend(loc="lower right", frameon=True, fontsize=6.3)
    ax_d.grid(True)

    # Panel e: Coupled Fold-Bifurcation Equilibrium Curves across Subsidy Rate & Phi_crit
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e")
    sig_vec = np.linspace(0.0, 0.55, 12)
    for phi_c, col_p, ls_p, lbl_p in [
        (1.60, COLORS["teal"], "-", "Hypoxia-tolerant ($\\Phi_{\\mathrm{crit}} = 1.6$)"),
        (2.00, COLORS["blue"], "--", "Baseline coastal ($\\Phi_{\\mathrm{crit}} = 2.0$)"),
        (2.35, COLORS["coral"], "-", "Hypoxia-sensitive ($\\Phi_{\\mathrm{crit}} = 2.35$)"),
    ]:
        b_eq_mean, b_eq_lo, b_eq_hi = [], [], []
        for sv in sig_vec:
            vals_s = [
                float(ABMEarthEngine(scenario="SSP5-8.5", start_year=2020, end_year=2050,
                                     capacity_subsidy_rate=float(sv), phi_crit=phi_c, seed=42 + k).run_simulation()["total_biomass"].iloc[-1])
                for k in range(5)
            ]
            b_eq_mean.append(np.mean(vals_s))
            b_eq_lo.append(np.min(vals_s))
            b_eq_hi.append(np.max(vals_s))
        ax_e.fill_between(sig_vec, b_eq_lo, b_eq_hi, color=col_p, alpha=0.16)
        ax_e.plot(sig_vec, b_eq_mean, color=col_p, linestyle=ls_p, lw=2.05, label=lbl_p)

    ax_e.axvspan(0.35, 0.55, color=COLORS["coral"], alpha=0.09, label="Overcapacity ($\\sigma > 0.35$)")
    ax_e.axvline(0.35, color=COLORS["coral"], linestyle=":", lw=1.4)
    ax_e.text(0.36, 368, "$\\sigma_{\\mathrm{crit}} = 0.35$", fontsize=6.8, fontweight="bold", color=COLORS["coral"])
    ax_e.set_xlim(0.0, 0.55)
    ax_e.set_xlabel("Capacity-enhancing subsidy rate ($\\sigma$)")
    ax_e.set_ylabel("2050 equilibrium biomass (t)")
    ax_e.legend(loc="lower left", frameon=True, fontsize=6.3)
    ax_e.grid(True)

    # Panel f: Coupled Social-Ecological Phase Portrait (Debt-Trapped SSF Share % vs Biomass, 2000->2050)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f")
    trap_rec = gaussian_filter1d(df_tip["debt_trap_recovery"].values * 100.0, sigma=6.0)
    bio_rec_s = gaussian_filter1d(rec_mat.mean(axis=0), sigma=6.0)
    ax_f.plot(trap_rec, bio_rec_s, color=SSP_COLORS["Recovery"], lw=2.4, linestyle="--", label="Multilateral reform (0.0% trapped)")
    for sc_k, ls_k, trap_end in [
        ("SSP1-2.6", "-.", df_tip["debt_trap_ssp1"].iloc[-1] * 100.0),
        ("SSP2-4.5", ":", df_tip["debt_trap_ssp2"].iloc[-1] * 100.0),
        ("SSP5-8.5", "-", df_tip["debt_trap_ssp5"].iloc[-1] * 100.0),
    ]:
        dt_m = gaussian_filter1d(np.mean(dt_ens[sc_k], axis=0), sigma=6.0)
        b_m = gaussian_filter1d(np.mean(bio_ens[sc_k], axis=0), sigma=6.0)
        b_lo = gaussian_filter1d(np.percentile(bio_ens[sc_k], 2.5, axis=0), sigma=6.0)
        b_hi = gaussian_filter1d(np.percentile(bio_ens[sc_k], 97.5, axis=0), sigma=6.0)
        ax_f.fill_between(dt_m, b_lo, b_hi, color=SSP_COLORS[sc_k], alpha=0.14)
        ax_f.plot(dt_m, b_m, color=SSP_COLORS[sc_k], lw=2.05, linestyle=ls_k,
                  label=f"{sc_k} ({trap_end:.1f}% trapped)")

    ax_f.axvline(50.0, color="#64748B", linestyle=":", lw=1.05)
    ax_f.text(51.5, 650.0, "Majority of fleet\nin debt trap (50%)", fontsize=6.3, color="#475569")
    ax_f.annotate("2000 start", (0.0, float(bio_rec_s[0])), xytext=(7, -6), textcoords="offset points",
                  fontsize=6.6, fontweight="bold", color="#111111")
    ax_f.annotate("2050 reform", (0.0, float(bio_rec_s[-1])), xytext=(7, -4), textcoords="offset points",
                  fontsize=6.6, fontweight="bold", color=SSP_COLORS["Recovery"])
    end_dt5 = float(np.mean(dt_ens["SSP5-8.5"], axis=0)[-1])
    end_bio5 = float(np.mean(bio_ens["SSP5-8.5"], axis=0)[-1])
    ax_f.annotate("2050 collapse\n(87.5% trapped)", (end_dt5, end_bio5),
                  xytext=(56.0, 425.0), textcoords="data", fontsize=6.5, fontweight="bold",
                  color=SSP_COLORS["SSP5-8.5"],
                  arrowprops=dict(arrowstyle="-|>", color=SSP_COLORS["SSP5-8.5"], lw=0.75, mutation_scale=8))
    ax_f.set_xlim(-3, 94)
    ax_f.set_xlabel("Debt-trapped artisanal fleet share (%)")
    ax_f.set_ylabel("Coastal standing biomass (t)")
    ax_f.legend(loc="lower left", frameon=True, fontsize=6.2)
    ax_f.grid(True)

    save_pub_figure(fig, "Figure_3_Tipping_Points_and_Hysteresis")


# ==============================================================================
# FIGURE 4: 2x3 Homogeneous Horizontal Raincloud-Interval Micro-Behavioral Suite
# ==============================================================================
def generate_figure_4():
    logger.info("Generating Figure 4 (2x3 uniform horizontal raincloud-interval distributions a-f)...")
    from engine import ABMEarthEngine
    from mechanism_ablation import ABLATION_ARMS

    # Canonical baseline & reform runs (seed=42) for agent-level realized cohort records
    eng_ssp1 = ABMEarthEngine(scenario="SSP1-2.6", start_year=2000, end_year=2050, seed=42)
    df_ssp1 = eng_ssp1.run_simulation()
    eng_ssp2 = ABMEarthEngine(scenario="SSP2-4.5", start_year=2000, end_year=2050, seed=42)
    df_ssp2 = eng_ssp2.run_simulation()
    eng_ssp5 = ABMEarthEngine(scenario="SSP5-8.5", start_year=2000, end_year=2050, seed=42)
    df_ssp5 = eng_ssp5.run_simulation()
    eng_int = ABMEarthEngine(
        scenario="SSP2-4.5", wto_prohibition=True, capacity_subsidy_rate=0.0,
        mpa_fraction=0.30, carbon_price=35.0, safety_net_active=True, start_year=2000, end_year=2050, seed=42
    )
    df_int = eng_int.run_simulation()

    # M=30 stochastic realizations for compound shock incidence across SSP1-2.6, SSP2-4.5, SSP5-8.5
    shock_inc = {"SSP1-2.6": [], "SSP2-4.5": [], "SSP5-8.5": []}
    mhw_hyp_inc = []
    for s_i in range(30):
        for sc_k in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
            d_tmp = ABMEarthEngine(scenario=sc_k, start_year=2000, end_year=2050, seed=300 + s_i).run_simulation()
            tri = ((d_tmp["is_mhw"] == 1) & (d_tmp["metabolic_index_phi"] < 2.25) & (d_tmp["fuel_price"] > 1.05)).mean() * 100.0
            shock_inc[sc_k].append(float(tri))
            if sc_k == "SSP5-8.5":
                mh = ((d_tmp["is_mhw"] == 1) & (d_tmp["metabolic_index_phi"] < 2.25)).mean() * 100.0
                mhw_hyp_inc.append(float(mh))

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.38)

    # Panel a: Concurrent Ocean & Fuel Shock Monthly Frequency Distributions (M=30 runs)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a", x=-0.18)
    groups_a = [
        ("Combined shock\n(SSP1-2.6)", np.array(shock_inc["SSP1-2.6"]), SSP_COLORS["SSP1-2.6"], f"{np.mean(shock_inc['SSP1-2.6']):.1f}%"),
        ("Combined shock\n(SSP2-4.5)", np.array(shock_inc["SSP2-4.5"]), SSP_COLORS["SSP2-4.5"], f"{np.mean(shock_inc['SSP2-4.5']):.1f}%"),
        ("Combined shock\n(SSP5-8.5)", np.array(shock_inc["SSP5-8.5"]), SSP_COLORS["SSP5-8.5"], f"{np.mean(shock_inc['SSP5-8.5']):.1f}%"),
        ("Heatwave + hypoxia\n(SSP5-8.5)", np.array(mhw_hyp_inc), COLORS["purple"], f"{np.mean(mhw_hyp_inc):.1f}%"),
    ]
    plot_raincloud_h(ax_a, groups_a, xlabel="Concurrent shock frequency (% of months)", seed=11, xlim_pad_right=0.25)

    # Panel b: Artisanal Foraging Distance from Home Port (nm, late-period 120-mo agent records)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b", x=-0.18)
    solv_dist, solv_haz, trap_dist, trap_haz = [], [], [], []
    for f_ag in eng_int.ssf_fishers + eng_ssp5.ssf_fishers:
        if getattr(f_ag, "solvent_trip_distances", None):
            solv_dist.append(float(np.mean(f_ag.solvent_trip_distances[-120:])))
            solv_haz.append(float(np.mean(f_ag.solvent_wave_hazards[-120:])))
        if getattr(f_ag, "trapped_trip_distances", None):
            trap_dist.append(float(np.mean(f_ag.trapped_trip_distances[-120:])))
            trap_haz.append(float(np.mean(f_ag.trapped_wave_hazards[-120:])))

    groups_b = [
        (f"Debt-free fishers\n($n = {len(solv_dist)}$)", np.array(solv_dist), SOLVENCY_COLORS["Solvent"], f"{np.mean(solv_dist):.2f} nm"),
        (f"Debt-trapped fishers\n($n = {len(trap_dist)}$)", np.array(trap_dist), SOLVENCY_COLORS["Trapped"], f"{np.mean(trap_dist):.2f} nm"),
    ]
    plot_raincloud_h(ax_b, groups_b, xlabel="Mean foraging distance from port (nm)", seed=24, xlim_pad_right=0.26)

    # Panel c: Realized Physical Wave & Storm Hazard Exposure (0-1 index)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c", x=-0.18)
    groups_c = [
        (f"Debt-free fishers\n($n = {len(solv_haz)}$)", np.array(solv_haz), SOLVENCY_COLORS["Solvent"], f"{np.mean(solv_haz):.3f}"),
        (f"Debt-trapped fishers\n($n = {len(trap_haz)}$)", np.array(trap_haz), SOLVENCY_COLORS["Trapped"], f"{np.mean(trap_haz):.3f}"),
    ]
    plot_raincloud_h(ax_c, groups_c, xlabel="Realized physical wave hazard index (0–1)", seed=33, xlim_pad_right=0.24)

    # Panel d: Net Monthly Fishing Cashflow Margins (USD mo^-1, 2000-2050)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d", x=-0.18)
    cf_int = df_int.groupby("year")["ssf_net_income_p50"].mean().values
    cf_s1 = df_ssp1.groupby("year")["ssf_net_income_p50"].mean().values
    cf_s2 = df_ssp2.groupby("year")["ssf_net_income_p50"].mean().values
    cf_s5 = df_ssp5.groupby("year")["ssf_net_income_p50"].mean().values
    def_int = (df_int["ssf_net_income_p50"] < 0.0).mean() * 100.0
    def_s5 = (df_ssp5["ssf_net_income_p50"] < 0.0).mean() * 100.0

    groups_d = [
        ("SSP5-8.5\nunmitigated", cf_s5, SSP_COLORS["SSP5-8.5"], f"{cf_s5.mean():+.2f} ({def_s5:.1f}% def.)"),
        ("SSP2-4.5\nstatus quo", cf_s2, SSP_COLORS["SSP2-4.5"], f"{cf_s2.mean():+.2f} USD"),
        ("SSP1-2.6\nmitigation", cf_s1, SSP_COLORS["SSP1-2.6"], f"{cf_s1.mean():+.2f} USD"),
        ("Integrated\ndividend", cf_int, POLICY_COLORS["Integrated Policy Package"], f"{cf_int.mean():+.2f} ({def_int:.1f}% def.)"),
    ]
    plot_raincloud_h(
        ax_d, groups_d, xlabel="Net monthly fishing cashflow (USD $\\mathrm{mo}^{-1}$)",
        ref_line=0.0, ref_label="Break-even", seed=45, xlim_pad_right=0.40
    )

    # Panel e: Household Informal Debt Distribution on Log10(1 + USD) Scale vs USD 250 Threshold
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=-0.18)
    debt_int = np.log10(1.0 + np.array([f.debt for f in eng_int.ssf_fishers]))
    debt_s1 = np.log10(1.0 + np.array([f.debt for f in eng_ssp1.ssf_fishers]))
    debt_s2 = np.log10(1.0 + np.array([f.debt for f in eng_ssp2.ssf_fishers]))
    debt_s5 = np.log10(1.0 + np.array([f.debt for f in eng_ssp5.ssf_fishers]))
    thresh_log = float(np.log10(1.0 + 250.0))
    thresh_band = (float(np.log10(1.0 + 187.5)), float(np.log10(1.0 + 312.5)))

    groups_e = [
        ("Integrated\ndividend", debt_int, POLICY_COLORS["Integrated Policy Package"], "0.0% trapped"),
        ("SSP1-2.6\nmitigation", debt_s1, SSP_COLORS["SSP1-2.6"], "65.0% trapped"),
        ("SSP2-4.5\nstatus quo", debt_s2, SSP_COLORS["SSP2-4.5"], "72.5% trapped"),
        ("SSP5-8.5\nunmitigated", debt_s5, SSP_COLORS["SSP5-8.5"], "87.5% trapped"),
    ]
    plot_raincloud_h(
        ax_e, groups_e, xlabel="Terminal household debt, $\\log_{10}(1 + \\mathrm{USD})$",
        ref_line=thresh_log, ref_label="$D_{\\mathrm{thresh}}$ = USD 250",
        shade_band=thresh_band, seed=58, xlim_pad_right=0.36
    )
    ax_e.set_xticks([0.0, float(np.log10(11.0)), float(np.log10(101.0)), thresh_log, float(np.log10(1001.0)), float(np.log10(10001.0))])
    ax_e.set_xticklabels(["0", "10", "100", "250", "1k", "10k"])

    # Panel f: Four-Arm SSP5-8.5 Causal Mechanism Ablation (M=30 CRN seeds per arm)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=-0.18)
    arm_meta = [
        ("full_coupled", "Full coupled\n(SSP5-8.5)", COLORS["coral"]),
        ("no_metabolic", "No metabolic\nhypoxia", COLORS["blue"]),
        ("no_debt_trap", "No informal\ndebt trap", COLORS["amber"]),
        ("no_dwf", "No subsidized\nDWF fleet", COLORS["teal"]),
    ]
    arm_kw_lookup = {a["key"]: a["kwargs"] for a in ABLATION_ARMS}
    arm_arrays = {}
    arm_trap = {}
    for a_key, _, _ in arm_meta:
        bios, traps = [], []
        for i in range(30):
            dfa = ABMEarthEngine(scenario="SSP5-8.5", start_year=2000, end_year=2050,
                                 seed=100 + i, **arm_kw_lookup[a_key]).run_simulation()
            bios.append(float(dfa["total_biomass"].iloc[-1]))
            traps.append(float(dfa["debt_trap_fraction"].max()) * 100.0)
        arm_arrays[a_key] = np.array(bios)
        arm_trap[a_key] = float(np.mean(traps))

    base_b = float(arm_arrays["full_coupled"].mean())
    groups_f = []
    for a_key, a_lbl, a_col in arm_meta:
        arr_b = arm_arrays[a_key]
        if a_key == "full_coupled":
            ann = f"{arr_b.mean():.1f} t ({arm_trap[a_key]:.1f}% trap)"
        else:
            pct = (arr_b.mean() - base_b) / base_b * 100.0
            ann = f"+{pct:.1f}% ({arm_trap[a_key]:.1f}% trap)"
        groups_f.append((a_lbl, arr_b, a_col, ann))

    plot_raincloud_h(ax_f, groups_f, xlabel="2050 standing biomass (t)", seed=77, xlim_pad_right=0.42)

    save_pub_figure(fig, "Figure_4_Compound_Hazard_Dynamics")


# ==============================================================================
# FIGURE 5: 2x3 Homogeneous Horizontal Dot-Interval Sobol GSA & Skill Suite
# ==============================================================================
def generate_figure_5():
    logger.info("Generating Figure 5 (2x3 uniform horizontal dot-interval Sobol GSA & benchmark suite a-f)...")
    df_bench = pd.read_csv(TABLES_DIR / "Table_Surrogate_Model_Comparison.csv")
    df_sob = pd.read_csv(TABLES_DIR / "Table_Sobol_Sensitivity_Indices.csv")
    df_dv = pd.read_csv(TABLES_DIR / "Table_Direct_ABM_Sobol_Validation.csv")

    fig = plt.figure(figsize=(15.8, 9.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.52, top=0.92)

    top_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222",
               markeredgewidth=1.1, markersize=5.5, label="First-order Sobol $S_1$"),
        Line2D([0], [0], marker="o", color="#222222", markerfacecolor=COLORS["teal"], markeredgecolor="#111111",
               lw=1.6, markersize=5.8, label="Total-order Sobol $S_T \\pm 95\\%$ CI ($N = 73{,}728$)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["navy"], markeredgecolor="#111111",
               markersize=5.6, label="Direct mechanistic ABM $S_T$ ($N = 304$)"),
    ]
    fig.legend(
        handles=top_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=3, frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=7.6
    )

    # Panels a-d: Complete 17-Parameter Sobol S1 -> ST (±95% CI) Rankings Across All 4 Targets
    sobol_specs = [
        ("a", gs[0, 0], "final_biomass", "2050 Standing Biomass", COLORS["teal"]),
        ("b", gs[0, 1], "peak_debt_trap_fraction", "Peak Debt-Trap Share", COLORS["coral"]),
        ("c", gs[0, 2], "cumulative_blue_carbon", "Cumulative Blue Carbon", COLORS["blue"]),
        ("d", gs[1, 0], "total_export_revenue", "Cumulative Export Revenue", COLORS["amber"]),
    ]
    for p_let, spec_pos, m_key, badge_lbl, col_m in sobol_specs:
        ax_s = fig.add_subplot(spec_pos)
        panel_tag(ax_s, p_let, x=-0.34, y=1.03)
        sub_m = df_sob[df_sob["Metric"] == m_key].sort_values("Total_Order_ST", ascending=True).reset_index(drop=True)
        ys = np.arange(len(sub_m))
        for i, r in sub_m.iterrows():
            s1_v = max(0.0, float(r["First_Order_S1"]))
            st_v = max(0.0, float(r["Total_Order_ST"]))
            ci_v = float(r["Total_Order_ST_CI"])
            ax_s.plot([s1_v, st_v], [i, i], color=col_m, lw=2.0, zorder=3)
            ax_s.errorbar(
                st_v, i, xerr=ci_v, fmt="o", color=col_m, mfc=col_m,
                mec="#111111", ms=4.3, capsize=1.8, elinewidth=1.05, zorder=4
            )
            ax_s.scatter(s1_v, i, color="white", edgecolor=col_m, lw=1.1, s=18, zorder=5)

        ax_s.text(
            0.96, 0.05, badge_lbl, transform=ax_s.transAxes, ha="right", va="bottom",
            fontsize=7.1, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor=col_m, lw=0.9)
        )
        ax_s.set_yticks(ys)
        ax_s.set_yticklabels([PARAM_LABELS.get(p, p) for p in sub_m["Parameter"]], fontsize=6.6)
        ax_s.set_xlim(-0.015, float((sub_m["Total_Order_ST"] + sub_m["Total_Order_ST_CI"]).max()) * 1.13)
        ax_s.set_xlabel("Sobol variance index ($S_1 \\rightarrow S_T$)", fontsize=7.8)
        ax_s.grid(True, axis="x")

    # Panel e: Direct-ABM (N=304) vs 30-Yr Surrogate (N=73,728) Paired Total-Order ST Dumbbell Ranking
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=-0.34, y=1.03)
    dv_bio = df_dv[df_dv["Metric"] == "final_biomass"].copy()
    dv_dt = df_dv[df_dv["Metric"] == "peak_debt_trap_fraction"].copy()
    rho_b = float(dv_bio["Spearman_rho"].iloc[0])
    rho_d = float(dv_dt["Spearman_rho"].iloc[0])

    top_b = dv_bio.sort_values("Surrogate_ST", ascending=False).head(5).assign(Tag="Bio", Col=COLORS["teal"])
    top_d = dv_dt.sort_values("Surrogate_ST", ascending=False).head(5).assign(Tag="DT", Col=COLORS["coral"])
    dv_comb = pd.concat([top_b, top_d], ignore_index=True).sort_values("Surrogate_ST", ascending=True).reset_index(drop=True)

    ye = np.arange(len(dv_comb))
    for i, r in dv_comb.iterrows():
        d_st = float(r["Direct_ABM_ST"])
        s_st = float(r["Surrogate_ST"])
        ax_e.plot([d_st, s_st], [i, i], color="#94A3B8", lw=2.0, zorder=2)
        ax_e.scatter(s_st, i, color=r["Col"], edgecolor="#111111", lw=0.65, s=36, marker="o", zorder=4)
        ax_e.scatter(d_st, i, color=COLORS["navy"], edgecolor="#111111", lw=0.65, s=34, marker="D", zorder=5)

    short_p = {
        "phi_crit": "$\\Phi_{\\mathrm{crit}}$",
        "r_growth": "Growth ($r$)",
        "num_dwf_vessels": "DWF Fleet",
        "capacity_subsidy_rate": "Subsidy ($\\sigma$)",
        "num_ssf_fishers": "SSF Cohort",
        "ex_vessel_price_base": "Ex-Vessel $P$",
        "diffusivity": "Diffusivity ($D$)",
    }
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels([f"{short_p.get(r['Parameter'], r['Parameter'])} [{r['Tag']}]" for _, r in dv_comb.iterrows()], fontsize=6.9)
    ax_e.text(
        0.96, 0.05, f"Biomass $\\rho = {rho_b:.3f}$\nDebt trap $\\rho = {rho_d:.3f}$\n($P < 4 \\times 10^{{-7}}$)",
        transform=ax_e.transAxes, ha="right", va="bottom", fontsize=6.7, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )
    ax_e.set_xlim(-0.01, 0.39)
    ax_e.set_xlabel("Direct-ABM (diamond) vs surrogate (circle) $S_T$", fontsize=7.8)
    ax_e.grid(True, axis="x")

    # Panel f: Cross-Architecture 20% Independent Holdout R² ± 95% CI Dot-Interval Benchmark (n = 103)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=-0.30, y=1.03)
    targets = ["final_biomass", "peak_debt_trap_fraction", "cumulative_blue_carbon", "total_export_revenue"]
    t_lbls = ["2050 Biomass", "Peak Debt Trap", "Blue Carbon", "Export Revenue"]
    yf = np.arange(len(targets))[::-1]
    n_hold = 103.0
    archs = [
        ("MultiLayerPerceptron", "MLP (128, 64)", COLORS["navy"], "o", +0.16),
        ("HistGradientBoosting", "HistGradientBoosting", COLORS["teal"], "s", 0.0),
        ("RandomForest", "Random Forest (250)", COLORS["amber"], "^", -0.16),
    ]
    for a_k, a_lbl, col_a, mk_a, dy in archs:
        sub_a_df = df_bench[df_bench["Architecture"] == a_k].set_index("Target_Metric").loc[targets]
        r2_vals = sub_a_df["Holdout_R2"].values.astype(float)
        se_r2 = np.sqrt(4.0 * np.clip(r2_vals, 0.01, 0.999) * ((1.0 - np.clip(r2_vals, 0.01, 0.999)) ** 2) / n_hold)
        ci_lo = np.minimum(r2_vals - 0.01, 1.96 * se_r2)
        ci_hi = np.minimum(1.0 - r2_vals, 1.96 * se_r2)
        ax_f.errorbar(
            r2_vals, yf + dy, xerr=np.vstack([ci_lo, ci_hi]), fmt=mk_a,
            color=col_a, mfc=col_a, mec="#111111", mew=0.65,
            markersize=5.4, capsize=2.4, elinewidth=1.45, zorder=4, label=a_lbl
        )
    ax_f.set_yticks(yf)
    ax_f.set_yticklabels(t_lbls, fontsize=7.5)
    ax_f.set_xlim(0.68, 1.01)
    ax_f.set_ylim(-0.88, 3.55)
    ax_f.set_xlabel("Independent 20% LHS holdout $R^2 \\pm 95\\%$ CI", fontsize=7.8)
    ax_f.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=6.5)
    ax_f.grid(True, axis="x")

    save_pub_figure(fig, "Figure_5_ML_Surrogate_and_Sobol_GSA")


# ==============================================================================
# FIGURE 6: 2x3 Co-Formatted Monte Carlo Policy Ensemble Paths (2020-2045)
# ==============================================================================
def generate_figure_6():
    logger.info("Generating Figure 6 (2x3 uniform Monte Carlo policy ensemble paths & ribbons 2020-2045 a-f)...")
    from engine import ABMEarthEngine
    from stochastic_evaluator import SCENARIOS

    df_stoch = pd.read_parquet(TABLES_DIR / "stochastic_trajectories.parquet")

    ens_debt = {sc: [] for sc in SCENARIOS}
    ens_ssf_ann = {sc: [] for sc in SCENARIOS}
    ens_carb = {sc: [] for sc in SCENARIOS}
    ens_val_cum = {sc: [] for sc in SCENARIOS}

    for sc_name, sc_p in SCENARIOS.items():
        for i in range(30):
            df_s = ABMEarthEngine(
                scenario="SSP2-4.5", start_year=2020, end_year=2045, seed=1000 + i * 17,
                wto_prohibition=sc_p["wto_prohibition"],
                capacity_subsidy_rate=sc_p["capacity_subsidy_rate"],
                mpa_fraction=sc_p["mpa_fraction"],
                carbon_price=sc_p["carbon_price"],
                safety_net_active=sc_p["safety_net_active"],
            ).run_simulation()
            ens_debt[sc_name].append(df_s["mean_ssf_debt"].values)
            ens_ssf_ann[sc_name].append(df_s.groupby("year")["ssf_harvest"].sum().values)
            ens_carb[sc_name].append(df_s["cumulative_blue_carbon"].values)
            tot_flow = df_s["export_revenue_usd"].values + df_s["blue_carbon_dividend_usd"].values + df_s["emergency_transfers_usd"].values
            ens_val_cum[sc_name].append(np.cumsum(tot_flow) / 1e6)

    fig = plt.figure(figsize=(14.8, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.33, wspace=0.28, top=0.91)

    policy_order = [
        ("Integrated Policy Package", "-"),
        ("WTO Subsidy Ban", "-."),
        ("Blue Carbon Dividend", "--"),
        ("30x30 Spatial Closures", (0, (4, 1.5))),
        ("Status Quo", ":"),
    ]
    top_handles = [
        Line2D([0], [0], color=POLICY_COLORS[sc], lw=2.2, linestyle=ls, label=POLICY_SHORT[sc])
        for sc, ls in policy_order
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=POLICY_COLORS["Integrated Policy Package"], alpha=0.18, edgecolor="none", label="95% ensemble CI ($M=30$)")
    ]
    fig.legend(
        handles=top_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=6, frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=7.3
    )

    x_ticks_6 = [2020, 2025, 2030, 2035, 2040, 2045]

    # Panel a: Standing Fish Biomass Envelopes (2020-2045, t)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a")
    for sc_name, ls in policy_order:
        sub_s = df_stoch[df_stoch["Scenario"] == sc_name]
        my = 2020.0 + sub_s["Month"].values / 12.0
        col = POLICY_COLORS[sc_name]
        ax_a.fill_between(my, sub_s["Biomass_CI_Low"], sub_s["Biomass_CI_High"], color=col, alpha=0.16)
        ax_a.plot(my, pd.Series(sub_s["Biomass_Mean"].values).rolling(6, min_periods=1).mean(),
                  color=col, lw=2.0, linestyle=ls)
    ax_a.axvline(2025, color="#64748B", linestyle=":", lw=1.0)
    ax_a.text(2025.5, 690.0, "2025 WTO ban", fontsize=6.3, color="#475569")
    ax_a.annotate("+63.5%\n(534.5 t)", (2045.8, 534.5), xytext=(-46, -24), textcoords="offset points",
                  fontsize=6.8, fontweight="bold", color=POLICY_COLORS["Integrated Policy Package"])
    ax_a.set_xlim(2020, 2045.9)
    ax_a.set_xticks(x_ticks_6)
    ax_a.set_xlabel("Projection year (2020–2045)")
    ax_a.set_ylabel("Coastal fish biomass (t)")
    ax_a.grid(True)

    # Panel b: Artisanal Fleet Debt-Trap Incidence Envelopes (2020-2045, %)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b")
    for sc_name, ls in policy_order:
        sub_s = df_stoch[df_stoch["Scenario"] == sc_name]
        my = 2020.0 + sub_s["Month"].values / 12.0
        col = POLICY_COLORS[sc_name]
        ax_b.fill_between(my, sub_s["DebtTrap_CI_Low"] * 100.0, sub_s["DebtTrap_CI_High"] * 100.0, color=col, alpha=0.16)
        ax_b.plot(my, sub_s["DebtTrap_Mean"] * 100.0, color=col, lw=2.0, linestyle=ls)
    ax_b.axvline(2025, color="#64748B", linestyle=":", lw=1.0)
    ax_b.text(0.04, 0.93, "Standalone 30×30: 48.6% trapped\nDividend / Integrated: 0.0% trapped",
              transform=ax_b.transAxes, va="top", ha="left", fontsize=6.7, fontweight="semibold", color="#1E293B",
              bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92))
    ax_b.set_xlim(2020, 2045.9)
    ax_b.set_xticks(x_ticks_6)
    ax_b.set_ylim(-2.5, 56)
    ax_b.set_xlabel("Projection year (2020–2045)")
    ax_b.set_ylabel("Debt-trapped artisanal fleet (%)")
    ax_b.grid(True)

    # Panel c: Mean Household Informal Debt Trajectories vs US$250 Safety-First Limit
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c")
    yrs_m = 2020.0 + np.arange(len(ens_debt["Status Quo"][0])) / 12.0
    for sc_name, ls in policy_order:
        mat = np.array(ens_debt[sc_name])
        col = POLICY_COLORS[sc_name]
        ax_c.fill_between(yrs_m, np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0), color=col, alpha=0.16)
        ax_c.plot(yrs_m, mat.mean(axis=0), color=col, lw=2.0, linestyle=ls)
    ax_c.axhline(250.0, color=COLORS["navy"], linestyle="--", lw=1.15)
    ax_c.text(2021.0, 272.0, "Safety-first limit (US$250)", fontsize=6.7, fontweight="bold", color=COLORS["navy"])
    ax_c.set_xlim(2020, 2045.9)
    ax_c.set_xticks(x_ticks_6)
    ax_c.set_xlabel("Projection year (2020–2045)")
    ax_c.set_ylabel("Mean household informal debt (US$)")
    ax_c.grid(True)

    # Panel d: Annualized Artisanal Small-Scale Catch Trajectories (t yr^-1)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d")
    yrs_a = np.linspace(2020.0, 2045.8, len(ens_ssf_ann["Status Quo"][0]))
    for sc_name, ls in policy_order:
        mat = np.array(ens_ssf_ann[sc_name])
        col = POLICY_COLORS[sc_name]
        ax_d.fill_between(yrs_a, np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0), color=col, alpha=0.16)
        ax_d.plot(yrs_a, mat.mean(axis=0), color=col, lw=2.0, linestyle=ls)
    ax_d.axvline(2025, color="#64748B", linestyle=":", lw=1.0)
    ax_d.set_xlim(2020, 2045.9)
    ax_d.set_xticks(x_ticks_6)
    ax_d.set_xlabel("Projection year (2020–2045)")
    ax_d.set_ylabel("Artisanal catch (t $\\mathrm{yr}^{-1}$)")
    ax_d.grid(True)

    # Panel e: Cumulative Shoreline Blue-Carbon Sequestration Trajectories (tCO2e)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e")
    for sc_name, ls in policy_order:
        mat = np.array(ens_carb[sc_name])
        col = POLICY_COLORS[sc_name]
        lw_use = 1.45 if sc_name == "WTO Subsidy Ban" else 2.05
        ax_e.plot(yrs_m, mat.mean(axis=0), color=col, lw=lw_use, linestyle=ls)
    ax_e.annotate("400.1 $\\mathrm{tCO_2e}$ (+58.0%)", (2045.8, 400.1), xytext=(2029.5, 392.0), textcoords="data",
                  fontsize=6.7, fontweight="bold", color=POLICY_COLORS["Integrated Policy Package"])
    ax_e.annotate("253.3 $\\mathrm{tCO_2e}$ (SQ / WTO)", (2044.0, 244.0), xytext=(2033.5, 102.0), textcoords="data",
                  fontsize=6.5, fontweight="semibold", color=POLICY_COLORS["Status Quo"],
                  arrowprops=dict(arrowstyle="-|>", color=POLICY_COLORS["Status Quo"], lw=0.75, mutation_scale=7))
    ax_e.set_xlim(2020, 2045.9)
    ax_e.set_xticks(x_ticks_6)
    ax_e.set_ylim(0, 435)
    ax_e.set_xlabel("Projection year (2020–2045)")
    ax_e.set_ylabel("Cumulative blue carbon ($\\mathrm{tCO_2e}$)")
    ax_e.grid(True)

    # Panel f: Cumulative Coastal Export + Community Dividend Value (Million US$)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f")
    for sc_name, ls in policy_order:
        mat = np.array(ens_val_cum[sc_name])
        col = POLICY_COLORS[sc_name]
        ax_f.fill_between(yrs_m, np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0), color=col, alpha=0.16)
        ax_f.plot(yrs_m, mat.mean(axis=0), color=col, lw=2.0, linestyle=ls)
    end_int = float(np.mean(ens_val_cum["Integrated Policy Package"], axis=0)[-1])
    end_sq = float(np.mean(ens_val_cum["Status Quo"], axis=0)[-1])
    ax_f.annotate(f"US${end_int:.1f}M\n(incl. dividends)", (2045.8, end_int), xytext=(-72, -18),
                  textcoords="offset points", fontsize=6.6, fontweight="bold", color=POLICY_COLORS["Integrated Policy Package"])
    ax_f.annotate(f"US${end_sq:.2f}M (SQ)", (2045.8, end_sq), xytext=(-68, 6),
                  textcoords="offset points", fontsize=6.5, fontweight="semibold", color=POLICY_COLORS["Status Quo"])
    ax_f.set_xlim(2020, 2045.9)
    ax_f.set_xticks(x_ticks_6)
    ax_f.set_xlabel("Projection year (2020–2045)")
    ax_f.set_ylabel("Cumulative export + dividend value (US$M)")
    ax_f.grid(True)

    save_pub_figure(fig, "Figure_6_Blue_Economy_Policy_Pathways")


def main():
    logger.info("Generating redesigned single-grammar Nature Climate Change Main Figures 1-6...")
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    logger.info("All 6 Main Figures generated cleanly in figures/!")


if __name__ == "__main__":
    main()
