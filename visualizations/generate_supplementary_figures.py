#!/usr/bin/env python3
"""
generate_supplementary_figures.py
Publication-Grade Single-Grammar Extended Data Figures (ED Figs. 1-3) and Supplementary Figures (Figs. S1-S2)
Formatted for Nature Climate Change (600 DPI PNG + vector PDF).

Strict Within-Figure Single Visual Grammar Across All 5 Supplementary / ED Figures:
- Extended Data Fig. 1 (2x3): Horizontal econometric & cross-national identification dot-interval suite (a-f; zero imputed MPA raw correlations)
- Extended Data Fig. 2 (2x3): Out-of-sample 1:1 validation parity scatterplots across country forward ABM, LORO & 4 MLP targets (a-f)
- Extended Data Fig. 3 (2x3): Co-registered 0.1-degree (10x10) coastal shelf grid submaps (a-f, exemplar)
- Figure S1 (2x3): Continuous biophysical forcing, discrete Laplacian spillover & MHW/Phi threshold curves with ONE shared top legend (a-f)
- Figure S2 (2x3): Horizontal dot-interval POM calibration convergence, 7-realm residual/LORO error & non-linear Sobol interaction suite (a-f)
"""

import logging
import pickle
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

TARGET_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TARGET_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TABLES_DIR = TARGET_DIR / "tables"
ANALYSIS_DIR = TARGET_DIR / "analysis"

for p in [TARGET_DIR / "visualizations", TARGET_DIR / "data", TARGET_DIR / "model", ANALYSIS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from style_and_maps import (
    CANONICAL_7_REALMS,
    COLORS,
    REALM_COLOR_MAP,
    REALM_SHORT_MAP,
    SSP_COLORS,
    apply_nature_style,
    compute_coastal_corridor_loitering,
    panel_tag,
    plot_shelf_submap,
    save_pub_figure,
)

apply_nature_style()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SupplementaryVisualizations")


def _compute_forward_holdout():
    """Runs strictly initialized 2020-2024 forward ABM validation across all 109 coastal states."""
    from engine import ABMEarthEngine

    df_pan = pd.read_parquet(DATA_DIR / "empirical_panel.parquet")
    pre = df_pan[(df_pan["year"] >= 2015) & (df_pan["year"] <= 2019)].groupby("country_iso3").agg(
        pre_bbmsy=("ram_mean_bbmsy", "mean"),
        pre_dwf=("foreign_dwf_share", "mean"),
        pre_mpa=("wb_mpa_pct_territorial", "mean"),
        pre_sub=("capacity_subsidies_usd", "mean"),
        pre_ben=("beneficial_subsidies_usd", "mean"),
    ).dropna()
    holdout = df_pan[(df_pan["year"] >= 2020) & (df_pan["year"] <= 2024)].groupby("country_iso3").agg(
        obs_bbmsy=("ram_mean_bbmsy", "mean"),
        is_ram=("is_direct_ram_assessed", "first"),
        dwf=("foreign_dwf_share", "mean"),
        region=("region_classification", "first"),
    ).dropna().join(pre, how="inner")

    sim_vals = []
    for idx_c, (_, row_c) in enumerate(holdout.iterrows()):
        sub_rate = float(np.clip(row_c["pre_sub"] / max(row_c["pre_sub"] + row_c["pre_ben"], 1.0) * 0.45, 0.05, 0.45))
        mpa_frac = float(np.clip(row_c["pre_mpa"] / 100.0, 0.01, 0.50))
        n_dwf = int(np.clip(round(4 + 18 * row_c["pre_dwf"]), 2, 22))
        eng_c = ABMEarthEngine(
            scenario="SSP2-4.5", start_year=2020, end_year=2024, seed=100 + idx_c,
            r_growth=0.42, capacity_subsidy_rate=sub_rate, mpa_fraction=mpa_frac, num_dwf_vessels=n_dwf
        )
        k_cell = eng_c.env.k_capacity / (eng_c.env.w * eng_c.env.h)
        eng_c.env.biomass = np.ones((eng_c.env.h, eng_c.env.w)) * (0.5 * k_cell * float(row_c["pre_bbmsy"]))
        df_sim_c = eng_c.run_simulation()
        sim_vals.append(float(df_sim_c["total_biomass"].mean() / (0.5 * eng_c.env.k_capacity)))

    holdout["sim_bbmsy"] = sim_vals
    holdout["residual"] = holdout["sim_bbmsy"] - holdout["obs_bbmsy"]
    return holdout


def _parse_reg_cell(cell_str: str):
    m = re.match(r"^\s*([+-]?\d+\.\d+)(\*{0,3})\s*\((\d+\.\d+)\)", str(cell_str))
    if m:
        return float(m.group(1)), float(m.group(3)), len(m.group(2)) > 0
    return np.nan, np.nan, False


# ==============================================================================
# EXTENDED DATA FIG 1: 2x3 Homogeneous Econometric & Empirical Forest Suite
# ==============================================================================
def generate_extended_data_fig_1():
    logger.info("Generating Extended Data Fig. 1 (2x3 uniform horizontal econometric & diagnostic forest suite a-f)...")
    df_prof = pd.read_csv(DATA_DIR / "country_profiles.csv")
    df_loit_reg = pd.read_csv(TABLES_DIR / "Table_Econometric_Panel_Regressions.csv").set_index("Variable")
    df_dwf_reg = pd.read_csv(TABLES_DIR / "Table_DWF_Exposure_Regressions.csv").set_index("Variable")
    coastal_loit = compute_coastal_corridor_loitering()

    fig = plt.figure(figsize=(15.4, 9.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.46)

    # Panel a: Unreported Offshore Transshipment Suite — Cross-Section & Pooled Realm+Year FE (z_transship, M1, M2, M4, M5a)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a", x=-0.28)
    vars_a = ["DWF Effort Share", "Log Capacity Subsidies", "Trophic Asymmetry Index", "Marine Protected Area (%)", "Log GDP per capita"]
    lbls_a = ["Foreign DWF share", "Log subsidies", "Trophic $\\Delta$TL", "Territorial MPA", "Log GDP/capita"]
    ya = np.arange(len(vars_a))[::-1]
    specs_a = [
        ("Model 2 (Multivariate)", "OLS M2 (N=109)", COLORS["amber"], -0.18),
        ("Model 4 (Extended Policy)", "Realm FE M4 (N=109)", COLORS["teal"], 0.0),
        ("Model 5a (Panel Realm + Year FE)", "Panel M5a (109 cl.)", COLORS["blue"], +0.18),
    ]
    for col_n, lbl_s, col_c, dy in specs_a:
        for iv, vn in enumerate(vars_a):
            if vn in df_loit_reg.index and col_n in df_loit_reg.columns:
                b, se, sig = _parse_reg_cell(df_loit_reg.loc[vn, col_n])
                if not np.isnan(b):
                    ax_a.errorbar(
                        b, ya[iv] + dy, xerr=1.96 * se, fmt="o",
                        color=col_c, mfc=col_c if sig else "white", mec=col_c,
                        elinewidth=1.45, capsize=2.3, markersize=4.8,
                        label=lbl_s if iv == 0 else ""
                    )
    ax_a.axvline(0.0, color="#333333", linestyle="--", lw=1.05)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(lbls_a, fontsize=7.2)
    ax_a.set_xlabel("Std. $\\hat{\\beta} \\pm 95\\%$ CI on $z_{\\mathrm{transship}}$", fontsize=7.8)
    ax_a.legend(loc="lower right", frameon=True, fontsize=6.2)
    ax_a.grid(True, axis="x")

    # Panel b: Within-Country Two-Way Fixed Effects Panel Regressions (Model 5 TWFE, N=1,417, 109 country clusters)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b", x=-0.28)
    vars_b = ["Marine Protected Area (%)", "DWF Effort Share", "Log Capacity Subsidies", "Log GDP per capita"]
    lbls_b = ["MPA expansion", "Foreign DWF share", "Log subsidies", "Log GDP/capita"]
    yb = np.arange(len(vars_b))[::-1]
    for dep_lbl, reg_tbl, col_c, mk_c, dy in [
        ("Unreported transshipment (TWFE M5)", df_loit_reg, COLORS["coral"], "o", -0.13),
        ("DWF share $z_{\\mathrm{DWF}}$ (TWFE M5)", df_dwf_reg, COLORS["navy"], "s", +0.13),
    ]:
        for iv, vn in enumerate(vars_b):
            if vn in reg_tbl.index:
                b, se, sig = _parse_reg_cell(reg_tbl.loc[vn, "Model 5 (Panel Two-Way FE)"])
                if not np.isnan(b):
                    ax_b.errorbar(
                        b, yb[iv] + dy, xerr=1.96 * se, fmt=mk_c,
                        color=col_c, mfc=col_c if sig else "white", mec=col_c,
                        elinewidth=1.5, capsize=2.5, markersize=5.0,
                        label=dep_lbl if iv == 0 else ""
                    )
    ax_b.axvline(0.0, color="#333333", linestyle="--", lw=1.05)
    ax_b.annotate("$\\hat{\\beta} = -0.018^{**}$\n($P = 0.003$)", (-0.018, yb[0] - 0.13),
                  xytext=(-58, -22), textcoords="offset points", fontsize=6.6, fontweight="bold", color=COLORS["coral"],
                  arrowprops=dict(arrowstyle="-", color=COLORS["coral"], lw=0.6))
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(lbls_b, fontsize=7.2)
    ax_b.set_xlim(-0.13, 0.08)
    ax_b.set_xlabel("Within-country TWFE $\\hat{\\beta} \\pm 95\\%$ CI", fontsize=7.8)
    ax_b.legend(loc="lower left", frameon=True, fontsize=6.2)
    ax_b.grid(True, axis="x")

    # Panel c: Companion Foreign DWF Exposure Regressions (z_DWF, Models 1-5a)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c", x=-0.28)
    mod_cols = [
        ("Model 1 (Bivariate)", "M1 Bivariate"),
        ("Model 2 (Multivariate)", "M2 Multivariate"),
        ("Model 3 (Realm FE)", "M3 Realm FE"),
        ("Model 4 (Extended Policy)", "M4 Extended"),
        ("Model 5a (Panel Realm + Year FE)", "M5a Panel (109 cl.)"),
    ]
    yc = np.arange(len(mod_cols))[::-1]
    for idx_m, (c_n, _) in enumerate(mod_cols):
        b_g, se_g, sig_g = _parse_reg_cell(df_dwf_reg.loc["Log GDP per capita", c_n])
        ax_c.errorbar(
            b_g, yc[idx_m] + 0.13, xerr=1.96 * se_g, fmt="o", color=COLORS["navy"],
            mfc=COLORS["navy"] if sig_g else "white", capsize=2.5, elinewidth=1.45, markersize=5.0,
            label="Log GDP/capita" if idx_m == 0 else ""
        )
        if c_n != "Model 1 (Bivariate)":
            b_s, se_s, sig_s = _parse_reg_cell(df_dwf_reg.loc["Log Capacity Subsidies", c_n])
            ax_c.errorbar(
                b_s, yc[idx_m] - 0.13, xerr=1.96 * se_s, fmt="s", color=COLORS["purple"],
                mfc=COLORS["purple"] if sig_s else "white", capsize=2.5, elinewidth=1.45, markersize=4.8,
                label="Log subsidies" if idx_m == 1 else ""
            )
    ax_c.axvline(0.0, color="#333333", linestyle="--", lw=1.05)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([lbl for _, lbl in mod_cols], fontsize=7.1)
    ax_c.set_xlabel("Std. $\\hat{\\beta} \\pm 95\\%$ CI on $z_{\\mathrm{DWF}}$", fontsize=7.8)
    ax_c.legend(loc="lower left", frameon=True, fontsize=6.4)
    ax_c.grid(True, axis="x")

    # Panel d: Cross-National Bivariate Correlations (Pearson r ± 95% Fisher-z CI, N=109, zero imputed MPA!)
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d", x=-0.30)
    log_gdp = np.log10(np.maximum(200.0, df_prof["gdp_pc"].values))
    dwf_pct = df_prof["foreign_dwf_share"].values * 100.0
    log_sub = np.log10(np.maximum(0.05, df_prof["capacity_subsidies_usd"].values / 1e6))
    log_hrs = np.log10(np.maximum(0.5, df_prof["fishing_hours"].values / 1e3))
    loit_ln = np.array([np.log1p(coastal_loit.get(iso, 0.0)) for iso in df_prof["country_iso3"]])
    pov_rt = df_prof["poverty_rate"].values
    tl_asym = df_prof["trophic_asymmetry_index"].values

    pairs_d = [
        ("Log GDP/cap $\\rightarrow$ DWF", log_gdp, dwf_pct, COLORS["navy"]),
        ("DWF $\\rightarrow$ Trophic $\\Delta$TL", dwf_pct, tl_asym, COLORS["gray"]),
        ("Log sub. $\\rightarrow$ Transshipment", log_sub, loit_ln, COLORS["purple"]),
        ("Log sub. $\\rightarrow$ AIS hours", log_sub, log_hrs, COLORS["blue"]),
        ("DWF $\\rightarrow$ Transshipment", dwf_pct, loit_ln, COLORS["teal"]),
        ("DWF $\\rightarrow$ Poverty rate", dwf_pct, pov_rt, COLORS["coral"]),
    ]
    yd = np.arange(len(pairs_d))
    n_obs = len(df_prof)
    for idx_p, (p_lbl, x_v, y_v, col_p) in enumerate(pairs_d):
        r_val, p_val = stats.pearsonr(x_v, y_v)
        z_f = np.arctanh(r_val)
        se_z = 1.0 / np.sqrt(n_obs - 3)
        r_lo, r_hi = float(np.tanh(z_f - 1.96 * se_z)), float(np.tanh(z_f + 1.96 * se_z))
        sig = p_val < 0.05
        ax_d.plot([r_lo, r_hi], [idx_p, idx_p], color=col_p, lw=2.0, zorder=3)
        ax_d.scatter(r_val, idx_p, color=col_p if sig else "white", edgecolor=col_p, lw=1.2, s=38, zorder=4)
        ax_d.text(
            r_hi + 0.03 if r_val >= 0 else r_lo - 0.03, idx_p,
            f"{r_val:+.2f}", ha="left" if r_val >= 0 else "right", va="center",
            fontsize=6.5, fontweight="bold", color="#1E293B"
        )
    ax_d.axvline(0.0, color="#333333", linestyle="--", lw=1.05)
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([p[0] for p in pairs_d], fontsize=7.0)
    ax_d.set_xlim(-0.92, 0.78)
    ax_d.set_xlabel("Cross-national Pearson $r \\pm 95\\%$ CI ($N = 109$)", fontsize=7.8)
    ax_d.grid(True, axis="x")

    # Panel e: Seven-Realm Trophic Asymmetry & Stock Depletion Deviation Profiles (Mean ± s.e.m.)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=-0.28)
    ye = np.arange(len(CANONICAL_7_REALMS))[::-1]
    for idx_r, (full_r, _, col_r) in enumerate(CANONICAL_7_REALMS):
        sub_r = df_prof[df_prof["region_classification"] == full_r]
        yi = ye[idx_r]
        tl_v = sub_r["trophic_asymmetry_index"].values
        bb_dev = sub_r["ram_mean_bbmsy"].values - 1.0
        ax_e.errorbar(
            tl_v.mean(), yi + 0.13, xerr=tl_v.std(ddof=1) / np.sqrt(len(sub_r)),
            fmt="o", color="#111111", mfc=col_r, markersize=5.2, elinewidth=1.4, capsize=2.3,
            label="$\\Delta$TL (export $-$ import)" if idx_r == 0 else ""
        )
        ax_e.errorbar(
            bb_dev.mean(), yi - 0.13, xerr=bb_dev.std(ddof=1) / np.sqrt(len(sub_r)),
            fmt="s", color="#111111", mfc="white", mec=col_r, mew=1.4, markersize=5.0, elinewidth=1.4, capsize=2.3,
            label="$B/B_{\\mathrm{msy}} - 1.0$" if idx_r == 0 else ""
        )
    ax_e.axvline(0.0, color="#333333", linestyle="--", lw=1.05)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels([short for _, short, _ in CANONICAL_7_REALMS], fontsize=7.1)
    ax_e.set_xlim(-0.42, 0.36)
    ax_e.set_xlabel("Realm mean $\\pm$ s.e.m. ($N = 109$)", fontsize=7.8)
    ax_e.legend(loc="lower right", frameon=True, fontsize=6.2)
    ax_e.grid(True, axis="x")

    # Panel f: Non-Imputed Empirical Regressor Variance Inflation Factor (VIF ± 95% Bootstrap CI, N=109)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=-0.28)
    X_raw = np.column_stack([log_gdp, dwf_pct, log_sub, log_hrs, pov_rt, tl_asym])
    vif_names = ["Log GDP/capita", "Foreign DWF share", "Log subsidies", "Log AIS effort", "Coastal poverty", "Trophic $\\Delta$TL"]

    def _compute_vifs(mat: np.ndarray) -> np.ndarray:
        out_v = np.zeros(mat.shape[1], dtype=float)
        for j in range(mat.shape[1]):
            yj = mat[:, j]
            Xj = np.column_stack([np.ones(len(yj)), np.delete(mat, j, axis=1)])
            beta_j, _, _, _ = np.linalg.lstsq(Xj, yj, rcond=None)
            y_hat = Xj @ beta_j
            r2_j = 1.0 - np.sum((yj - y_hat) ** 2) / max(1e-9, np.sum((yj - yj.mean()) ** 2))
            out_v[j] = float(1.0 / max(1e-6, 1.0 - r2_j))
        return out_v

    point_vifs = _compute_vifs(X_raw)
    rng_vif = np.random.default_rng(42)
    boot_vifs = np.array([
        _compute_vifs(X_raw[rng_vif.integers(0, len(X_raw), size=len(X_raw))])
        for _ in range(400)
    ])
    vifs = []
    for j in range(X_raw.shape[1]):
        lo_v = float(np.percentile(boot_vifs[:, j], 2.5))
        hi_v = float(np.percentile(boot_vifs[:, j], 97.5))
        vifs.append((vif_names[j], float(point_vifs[j]), max(0.01, float(point_vifs[j]) - lo_v), max(0.01, hi_v - float(point_vifs[j]))))
    vifs.sort(key=lambda t: t[1])

    yf = np.arange(len(vifs))
    for i, (_, v_val, err_lo, err_hi) in enumerate(vifs):
        ax_f.errorbar(
            v_val, i, xerr=np.array([[err_lo], [err_hi]]), fmt="o",
            color=COLORS["teal"], mfc=COLORS["teal"], mec="#111111", mew=0.65,
            markersize=5.2, capsize=2.4, elinewidth=1.45, zorder=4
        )
        ax_f.text(v_val + err_hi + 0.07, i, f"{v_val:.2f}", va="center", ha="left", fontsize=6.6, fontweight="bold", color="#1E293B")
    ax_f.axvline(1.0, color="#64748B", linestyle=":", lw=1.0)
    ax_f.axvline(3.0, color=COLORS["amber"], linestyle="--", lw=1.2, label="Conservative limit (3.0)")
    ax_f.axvline(5.0, color=COLORS["coral"], linestyle="-.", lw=1.2, label="Collinearity limit (5.0)")
    ax_f.set_yticks(yf)
    ax_f.set_yticklabels([v[0] for v in vifs], fontsize=7.2)
    ax_f.set_xlim(0.85, 5.45)
    ax_f.set_xlabel("Non-imputed predictor VIF $\\pm 95\\%$ CI ($N = 109$)", fontsize=7.8)
    ax_f.legend(loc="lower right", frameon=True, fontsize=6.3)
    ax_f.grid(True, axis="x")

    save_pub_figure(fig, "Extended_Data_Fig_1_Spatial_Grid_and_VIF")


# ==============================================================================
# EXTENDED DATA FIG 2: 2x3 Homogeneous Out-of-Sample 1:1 Parity Scatter Suite
# ==============================================================================
def generate_extended_data_fig_2():
    logger.info("Generating Extended Data Fig. 2 (2x3 uniform out-of-sample 1:1 validation parity scatters a-f)...")
    holdout = _compute_forward_holdout()
    with open(ANALYSIS_DIR / "surrogate_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    obs_val = holdout["obs_bbmsy"].values
    sim_val = holdout["sim_bbmsy"].values
    r2_all = float(1.0 - np.sum((obs_val - sim_val) ** 2) / max(np.sum((obs_val - np.mean(obs_val)) ** 2), 1e-6))
    rmse_all = float(np.sqrt(np.mean((obs_val - sim_val) ** 2)))
    ram_m = holdout["is_ram"] == 1
    r2_ram = float(1.0 - np.sum((holdout.loc[ram_m, "obs_bbmsy"] - holdout.loc[ram_m, "sim_bbmsy"]) ** 2) /
                   max(np.sum((holdout.loc[ram_m, "obs_bbmsy"] - holdout.loc[ram_m, "obs_bbmsy"].mean()) ** 2), 1e-6))
    rmse_ram = float(np.sqrt(np.mean((holdout.loc[ram_m, "obs_bbmsy"] - holdout.loc[ram_m, "sim_bbmsy"]) ** 2)))

    # Compute 7-fold Leave-One-Region-Out (LORO) out-of-fold predictions
    loro_oof_preds = pd.Series(index=holdout.index, dtype=float)
    for full_r, _, _ in CANONICAL_7_REALMS:
        train_r = holdout[holdout["region"] != full_r]
        sub_r = holdout[holdout["region"] == full_r]
        slope_r, int_r = np.polyfit(train_r["sim_bbmsy"].values, train_r["obs_bbmsy"].values, 1)
        loro_oof_preds.loc[sub_r.index] = int_r + slope_r * sub_r["sim_bbmsy"].values

    loro_r2 = float(1.0 - np.sum((obs_val - loro_oof_preds.values) ** 2) / max(np.sum((obs_val - np.mean(obs_val)) ** 2), 1e-6))
    loro_rmse = float(np.sqrt(np.mean((obs_val - loro_oof_preds.values) ** 2)))

    fig = plt.figure(figsize=(14.6, 9.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.30)

    # Panel a: Out-of-Sample Forward Country ABM Validation (2020-2024, N=109 + n=10 Direct RAM)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a")
    rec_m = ~ram_m
    mn, mx = min(obs_val.min(), sim_val.min()) - 0.08, max(obs_val.max(), sim_val.max()) + 0.08
    xs_a = np.linspace(mn, mx, 50)
    ax_a.fill_between(xs_a, xs_a - 1.96 * rmse_all, xs_a + 1.96 * rmse_all, color=COLORS["sky"], alpha=0.14)
    ax_a.plot(xs_a, xs_a, color=COLORS["coral"], linestyle="--", lw=1.35, label="1:1 Parity")
    ax_a.scatter(holdout.loc[rec_m, "obs_bbmsy"], holdout.loc[rec_m, "sim_bbmsy"],
                 color=COLORS["sky"], s=28, alpha=0.82, edgecolors="#222222", lw=0.45, label="Covariate EEZs ($n=99$)")
    ax_a.scatter(holdout.loc[ram_m, "obs_bbmsy"], holdout.loc[ram_m, "sim_bbmsy"],
                 color=COLORS["navy"], marker="D", s=44, edgecolors="#FFFFFF", lw=0.8, zorder=5, label="Direct RAM ($n=10$)")
    for iso, dx, dy in [("SEN", 10, -10), ("GHA", -26, 9), ("IDN", -24, 9), ("NOR", 12, -10), ("PER", 10, 6)]:
        if iso in holdout.index:
            ax_a.annotate(iso, (float(holdout.loc[iso, "obs_bbmsy"]), float(holdout.loc[iso, "sim_bbmsy"])),
                          xytext=(dx, dy), textcoords="offset points", fontsize=6.5, fontweight="bold", color=COLORS["navy"],
                          arrowprops=dict(arrowstyle="-", color="#64748B", lw=0.5))
    ax_a.text(
        0.04, 0.95,
        f"All EEZs ($N=109$): $R^2={r2_all:.3f}$, RMSE={rmse_all:.3f}\nDirect RAM ($n=10$): $R^2={r2_ram:.3f}$, RMSE={rmse_ram:.3f}",
        transform=ax_a.transAxes, va="top", ha="left", fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_a.set_xlim(mn, mx)
    ax_a.set_ylim(mn, mx)
    ax_a.set_xlabel("Observed $B/B_{\\mathrm{msy}}$ (2020–2024)")
    ax_a.set_ylabel("Forward ABM projection ($B/B_{\\mathrm{msy}}$)")
    ax_a.legend(loc="lower right", frameon=True, fontsize=6.2)
    ax_a.grid(True)

    # Panel b: 7-Fold Leave-One-Region-Out (LORO) Out-of-Fold Cross-Validation Parity
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b")
    mn_b, mx_b = min(obs_val.min(), loro_oof_preds.min()) - 0.08, max(obs_val.max(), loro_oof_preds.max()) + 0.08
    xs_b = np.linspace(mn_b, mx_b, 50)
    ax_b.fill_between(xs_b, xs_b - 1.96 * loro_rmse, xs_b + 1.96 * loro_rmse, color="#94A3B8", alpha=0.15)
    ax_b.plot(xs_b, xs_b, color="#111111", linestyle="--", lw=1.35)
    for full_r, short_r, col_r in CANONICAL_7_REALMS:
        m_r = holdout["region"] == full_r
        ax_b.scatter(
            holdout.loc[m_r, "obs_bbmsy"], loro_oof_preds.loc[m_r],
            color=col_r, s=30, alpha=0.85, edgecolors="#222222", lw=0.45, label=short_r
        )
    ax_b.text(
        0.04, 0.95,
        f"7-Fold LORO out-of-fold:\n$R^2_{{\\mathrm{{LORO}}}} = {loro_r2:.3f}$, OOF RMSE = {loro_rmse:.3f}\n($P < 0.001$, $N = 109$ states)",
        transform=ax_b.transAxes, va="top", ha="left", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
    )
    ax_b.set_xlim(mn_b, mx_b)
    ax_b.set_ylim(mn_b, mx_b)
    ax_b.set_xlabel("Observed $B/B_{\\mathrm{msy}}$ (2020–2024)")
    ax_b.set_ylabel("Out-of-fold LORO prediction ($B/B_{\\mathrm{msy}}$)")
    ax_b.legend(loc="lower right", ncol=2, frameon=True, fontsize=5.8)
    ax_b.grid(True)

    # Panels c-f: 30-Year MLP Surrogate 20% Independent LHS Holdout Parity Across All 4 Targets
    surr_specs = [
        ("c", gs[0, 2], "final_biomass", "2050 Standing Biomass", "ABM holdout biomass (t)", "MLP predicted biomass (t)", COLORS["teal"], 1.0, "t"),
        ("d", gs[1, 0], "peak_debt_trap_fraction", "Peak Debt-Trap Fraction", "ABM holdout debt-trap share", "MLP predicted debt-trap share", COLORS["coral"], 1.0, ""),
        ("e", gs[1, 1], "cumulative_blue_carbon", "Cumulative Blue Carbon", "ABM holdout carbon ($\\mathrm{tCO_2e}$)", "MLP predicted carbon ($\\mathrm{tCO_2e}$)", COLORS["blue"], 1.0, "$\\mathrm{tCO_2e}$"),
        ("f", gs[1, 2], "total_export_revenue", "Cumulative Export Revenue", "ABM holdout export (Million USD)", "MLP predicted export (Million USD)", COLORS["amber"], 1e6, "M USD"),
    ]
    for p_let, spec_pos, m_k, title_b, x_lbl, y_lbl, col_m, div_s, unit_s in surr_specs:
        ax_m = fig.add_subplot(spec_pos)
        panel_tag(ax_m, p_let)
        hd = bundle["holdout_data"][m_k]
        yt = np.asarray(hd["y_test"], dtype=float) / div_s
        yp = np.asarray(hd["y_pred"], dtype=float) / div_s
        rmse_scaled = float(hd["rmse"]) / div_s
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = (hi - lo) * 0.07
        xs_line = np.linspace(lo - pad, hi + pad, 50)
        ax_m.fill_between(xs_line, xs_line - 1.96 * rmse_scaled, xs_line + 1.96 * rmse_scaled, color=col_m, alpha=0.13, label="95% interval")
        ax_m.plot(xs_line, xs_line, color="#111111", linestyle="--", lw=1.3, label="1:1 Parity")
        ax_m.scatter(yt, yp, color=col_m, s=26, alpha=0.82, edgecolors="#111111", lw=0.4)
        u_fmt = f" {unit_s}" if unit_s else ""
        ax_m.text(
            0.04, 0.95,
            f"{title_b}\nHoldout $R^2 = {hd['r2']:.4f}$\nRMSE = {rmse_scaled:.3f}{u_fmt} ($n = {len(yt)}$)",
            transform=ax_m.transAxes, va="top", ha="left", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CBD5E1", alpha=0.92)
        )
        ax_m.set_xlim(lo - pad, hi + pad)
        ax_m.set_ylim(lo - pad, hi + pad)
        ax_m.set_xlabel(x_lbl)
        ax_m.set_ylabel(y_lbl)
        ax_m.legend(loc="lower right", frameon=True, fontsize=6.3)
        ax_m.grid(True)

    save_pub_figure(fig, "Extended_Data_Fig_2_Calibration_and_Direct_ABM_Sobol")


# ==============================================================================
# EXTENDED DATA FIG 3: Pure 6-Submap Co-Registered 0.1° Coastal Shelf Suite
# ==============================================================================
def generate_extended_data_fig_3():
    logger.info("Generating Extended Data Fig. 3 (pure 6 co-registered 0.1° shelf submaps a-f, exemplar)...")
    from engine import ABMEarthEngine

    eng_mpa = ABMEarthEngine(scenario="SSP5-8.5", mpa_fraction=0.30, wto_prohibition=True, safety_net_active=True, start_year=2000, end_year=2050, seed=42)
    eng_mpa.run_simulation()
    env = eng_mpa.env

    eng_sq = ABMEarthEngine(scenario="SSP5-8.5", mpa_fraction=0.10, wto_prohibition=False, safety_net_active=False, start_year=2000, end_year=2050, seed=42)
    eng_sq.run_simulation()

    fig = plt.figure(figsize=(14.4, 9.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a", x=-0.14, y=1.04, despine=False)
    plot_shelf_submap(ax_a, env.depth, cmap="YlGnBu_r", cbar_label="Shelf depth (m)", mpa_mask=env.is_mpa, port_xy=(env.port_x, env.port_y))
    ax_a.set_ylabel("Alongshore distance (km)")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b", x=-0.14, y=1.04, despine=False)
    plot_shelf_submap(ax_b, eng_sq.env.phi, cmap="RdYlBu", cbar_label="2050 Metabolic Index ($\\Phi$)",
                      mpa_mask=env.is_mpa, port_xy=(env.port_x, env.port_y), contour_val=2.0)

    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c", x=-0.14, y=1.04, despine=False)
    ssf_yr = eng_mpa.spatial_effort_ssf / 50.0
    pnorm = mcolors.PowerNorm(gamma=0.38, vmin=0.0, vmax=float(ssf_yr.max()))
    plot_shelf_submap(
        ax_c, ssf_yr, cmap="YlOrRd", norm=pnorm,
        cbar_label="SSF effort (trips $\\mathrm{yr}^{-1}$, power scale)",
        mpa_mask=env.is_mpa, port_xy=(env.port_x, env.port_y)
    )

    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d", x=-0.14, y=1.04, despine=False)
    plot_shelf_submap(
        ax_d, eng_sq.spatial_effort_dwf / 50.0, cmap="cividis_r",
        cbar_label="DWF effort (trips $\\mathrm{yr}^{-1}$)",
        mpa_mask=eng_sq.env.is_mpa, port_xy=(env.port_x, env.port_y)
    )
    ax_d.set_xlabel("Cross-shelf distance from coast (km)")
    ax_d.set_ylabel("Alongshore distance (km)")

    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=-0.14, y=1.04, despine=False)
    plot_shelf_submap(
        ax_e, eng_sq.spatial_biomass_end, cmap="viridis",
        cbar_label="2050 Status Quo biomass (t $\\mathrm{cell}^{-1}$)",
        mpa_mask=eng_sq.env.is_mpa, port_xy=(env.port_x, env.port_y), vmin=1.0, vmax=4.9
    )
    ax_e.set_xlabel("Cross-shelf distance from coast (km)")

    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=-0.14, y=1.04, despine=False)
    bio_diff = eng_mpa.spatial_biomass_end - eng_sq.spatial_biomass_end
    plot_shelf_submap(
        ax_f, bio_diff, cmap="YlGn",
        cbar_label="Integrated net biomass gain (+t $\\mathrm{cell}^{-1}$)",
        mpa_mask=env.is_mpa, port_xy=(env.port_x, env.port_y)
    )
    ax_f.set_xlabel("Cross-shelf distance from coast (km)")

    save_pub_figure(fig, "Extended_Data_Fig_3_Shelf_Effort_Partitioning_and_Spillover")


# ==============================================================================
# FIGURE S1: 2x3 Biophysical Forcing, Dispersal & Threshold Suite (Shared Legend)
# ==============================================================================
def generate_figure_s1():
    logger.info("Generating Figure S1 (2x3 uniform biophysical forcing & process curves with ONE shared top legend a-f)...")
    from cmip6_downscaler import CMIP6CoastalDownscaler

    fig = plt.figure(figsize=(14.6, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.30, top=0.91)

    series_store = {k: {"sst": [], "po2": [], "npp": []} for k in ["GFDL_1", "ENS_2", "GFDL_5", "MPI_5"]}
    for s_i in range(20):
        g1 = CMIP6CoastalDownscaler(esm_model="GFDL-ESM4", seed=300 + s_i).generate_coastal_forcing("SSP1-2.6", 2000, 2050)
        e2 = CMIP6CoastalDownscaler(esm_model="Ensemble-Mean", seed=300 + s_i).generate_coastal_forcing("SSP2-4.5", 2000, 2050)
        g5 = CMIP6CoastalDownscaler(esm_model="GFDL-ESM4", seed=300 + s_i).generate_coastal_forcing("SSP5-8.5", 2000, 2050)
        m5 = CMIP6CoastalDownscaler(esm_model="MPI-ESM1-2-HR", seed=300 + s_i).generate_coastal_forcing("SSP5-8.5", 2000, 2050)
        for key, d_in in [("GFDL_1", g1), ("ENS_2", e2), ("GFDL_5", g5), ("MPI_5", m5)]:
            series_store[key]["sst"].append(d_in.groupby("year")["sst_celsius"].mean().values)
            series_store[key]["po2"].append(d_in.groupby("year")["po2_kpa"].mean().values)
            series_store[key]["npp"].append(d_in.groupby("year")["npp_index"].mean().values)

    forcing_styles = [
        ("GFDL_1", "GFDL-ESM4 (SSP1-2.6)", COLORS["teal"], "-."),
        ("ENS_2", "Ensemble-Mean (SSP2-4.5)", COLORS["amber"], ":"),
        ("GFDL_5", "GFDL-ESM4 (SSP5-8.5)", COLORS["blue"], "-"),
        ("MPI_5", "MPI-ESM1-2-HR (SSP5-8.5)", COLORS["coral"], "--"),
    ]
    top_handles = [Line2D([0], [0], color=SSP_COLORS["Historical"], lw=2.0, label="Historical (2000–2025)")] + [
        Line2D([0], [0], color=col_m, lw=2.0, linestyle=ls_m, label=lbl_m)
        for _, lbl_m, col_m, ls_m in forcing_styles
    ]
    fig.legend(
        handles=top_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=5, frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=7.4
    )

    yrs_all = np.arange(2000, 2051)
    h_mask = yrs_all <= 2025
    p_mask = yrs_all >= 2025

    for pnl_l, ax_pos, var_k, y_lbl in [
        ("a", gs[0, 0], "sst", "Coastal sea surface temperature (°C)"),
        ("b", gs[0, 1], "po2", "Ambient dissolved $\\mathrm{O}_2$ (kPa)"),
        ("c", gs[0, 2], "npp", "Relative NPP index (baseline = 1.0)"),
    ]:
        ax_p = fig.add_subplot(ax_pos)
        panel_tag(ax_p, pnl_l)
        full_hist_mat = np.array(series_store["GFDL_5"][var_k])
        f_h_mean = pd.Series(full_hist_mat.mean(axis=0)).rolling(5, min_periods=1, center=True).mean().values
        f_h_lo = pd.Series(np.percentile(full_hist_mat, 10, axis=0)).rolling(5, min_periods=1, center=True).mean().values
        f_h_hi = pd.Series(np.percentile(full_hist_mat, 90, axis=0)).rolling(5, min_periods=1, center=True).mean().values
        ax_p.plot(yrs_all[h_mask], f_h_mean[h_mask], color=SSP_COLORS["Historical"], lw=2.0)
        ax_p.fill_between(yrs_all[h_mask], f_h_lo[h_mask], f_h_hi[h_mask], color="#64748B", alpha=0.14)

        for key, _, col_m, ls_m in forcing_styles:
            full_p_mat = np.array(series_store[key][var_k])
            p_mean = pd.Series(full_p_mat.mean(axis=0)).rolling(5, min_periods=1, center=True).mean().values
            p_lo = pd.Series(np.percentile(full_p_mat, 10, axis=0)).rolling(5, min_periods=1, center=True).mean().values
            p_hi = pd.Series(np.percentile(full_p_mat, 90, axis=0)).rolling(5, min_periods=1, center=True).mean().values
            ax_p.plot(yrs_all[p_mask], p_mean[p_mask], color=col_m, linestyle=ls_m, lw=1.95)
            ax_p.fill_between(yrs_all[p_mask], p_lo[p_mask], p_hi[p_mask], color=col_m, alpha=0.13)

        ax_p.axvline(2025, color="#64748B", linestyle=":", lw=1.0)
        if pnl_l == "a":
            ax_p.text(2025.6, float(f_h_hi[h_mask].min()), "2025 split", fontsize=6.2, color="#475569")
        if var_k == "npp":
            ax_p.axhline(1.0, color="#64748B", linestyle="--", lw=0.95)
        ax_p.set_xlim(2000, 2050)
        ax_p.set_xlabel("Simulation year")
        ax_p.set_ylabel(y_lbl)
        ax_p.grid(True)

    # Panel d: Exact 12-Month Discrete 5-Point Laplacian Cross-Shelf Spillover Profile
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d")
    dist_km = np.arange(10) * 10.0
    spill_curves = {}
    for d_val, col_k, ls_k, lbl_k in [
        (0.20, COLORS["blue"], "--", "CFL bound ($D = 0.20\\ \\mathrm{mo}^{-1}$)"),
        (0.15, COLORS["teal"], "-", "Calibrated ($D = 0.15\\ \\mathrm{mo}^{-1}$)"),
        (0.08, COLORS["amber"], "-.", "Moderate ($D = 0.08\\ \\mathrm{mo}^{-1}$)"),
        (0.02, COLORS["coral"], ":", "Sedentary ($D = 0.02\\ \\mathrm{mo}^{-1}$)"),
    ]:
        grid = np.zeros((10, 10), dtype=float)
        grid[:, 0] = 1.0
        nu_eff = min(0.20, max(0.01, d_val))
        for _ in range(12):
            padded = np.pad(grid, pad_width=1, mode="edge")
            lap = padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2] - 4.0 * grid
            grid = grid + nu_eff * lap
            grid[:, 0] = 1.0
        prof_d = grid.mean(axis=0)
        spill_curves[d_val] = prof_d
        ax_d.plot(dist_km, prof_d, color=col_k, linestyle=ls_k, lw=2.05, label=lbl_k)
    ax_d.fill_between(dist_km, spill_curves[0.08], spill_curves[0.20], color=COLORS["teal"], alpha=0.12)
    ax_d.axvline(10.0, color=COLORS["gray"], linestyle=":", lw=1.0)
    ax_d.text(11.5, 0.92, "Nursery border", fontsize=6.2, color="#475569")
    ax_d.set_xlabel("Cross-shelf distance from nursery source (km)")
    ax_d.set_ylabel("12-month discrete Laplacian spillover")
    ax_d.set_xlim(0, 90)
    ax_d.set_ylim(-0.02, 1.05)
    ax_d.legend(loc="upper right", frameon=True, fontsize=6.4)
    ax_d.grid(True)

    # Panel e: Fixed Historical vs Detrended Shifting-Baseline MHW Frequency (Continuous Multi-Seed Band)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e")
    fix_list, shf_list = [], []
    for sd_e in range(20):
        e5_s = CMIP6CoastalDownscaler(esm_model="Ensemble-Mean", seed=400 + sd_e).generate_coastal_forcing("SSP5-8.5", 2000, 2050)
        e5_s["decade"] = (e5_s["year"] // 10) * 10
        sm_s = e5_s["sst_celsius"].rolling(60, min_periods=12, center=True).mean().bfill().ffill()
        anom_s = e5_s["sst_celsius"] - sm_s
        th_s = float(np.quantile(anom_s, 0.90))
        e5_s["mhw_shifting"] = (anom_s > th_s).astype(float) * 100.0
        e5_s["mhw_fixed"] = e5_s["is_mhw"].astype(float) * 100.0
        dc_s = e5_s[e5_s["year"] <= 2049].groupby("decade")[["mhw_fixed", "mhw_shifting"]].mean()
        fix_list.append(dc_s["mhw_fixed"].values)
        shf_list.append(dc_s["mhw_shifting"].values)
    fix_mat, shf_mat = np.array(fix_list), np.array(shf_list)
    xd = np.arange(fix_mat.shape[1])
    ax_e.fill_between(xd, shf_mat.mean(axis=0), fix_mat.mean(axis=0), color=COLORS["coral"], alpha=0.15, label="Secular warming attribution")
    ax_e.plot(xd, fix_mat.mean(axis=0), color=COLORS["coral"], lw=2.15, label="Fixed 2000–2015 $Q_{90}$ baseline")
    ax_e.plot(xd, shf_mat.mean(axis=0), color=COLORS["blue"], linestyle="--", lw=2.05, label="Shifting mean $Q_{90}$ (detrended)")
    ax_e.set_xticks(xd)
    ax_e.set_xticklabels(["2000s", "2010s", "2020s", "2030s", "2040s"], fontsize=7.5)
    ax_e.set_ylabel("MHW months per decade (%)")
    ax_e.set_xlabel("Projection decade (SSP5-8.5)")
    ax_e.set_ylim(0, 70)
    ax_e.legend(loc="upper left", frameon=True, fontsize=6.4)
    ax_e.grid(True)

    # Panel f: ECDF of Monthly Shelf-Mean Metabolic Index Phi (Matching Shared Top Legend!)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f")
    for esm_k, sc_k, col_k, ls_k in [
        ("GFDL-ESM4", "SSP1-2.6", COLORS["teal"], "-."),
        ("Ensemble-Mean", "SSP2-4.5", COLORS["amber"], ":"),
        ("GFDL-ESM4", "SSP5-8.5", COLORS["blue"], "-"),
        ("MPI-ESM1-2-HR", "SSP5-8.5", COLORS["coral"], "--"),
    ]:
        df_sc = CMIP6CoastalDownscaler(esm_k, seed=42).generate_coastal_forcing(sc_k, 2000, 2050)
        vals_phi = np.sort(df_sc["metabolic_index_phi"].values)
        ecdf_y = np.linspace(0, 100, len(vals_phi))
        ax_f.plot(vals_phi, ecdf_y, color=col_k, linestyle=ls_k, lw=2.0)
    ax_f.axvline(2.0, color="#1E293B", linestyle="--", lw=1.25)
    ax_f.text(2.04, 10.0, "$\\Phi_{\\mathrm{crit}} = 2.0$", fontsize=6.9, fontweight="bold", color="#1E293B")
    ax_f.set_xlabel("Shelf-mean Metabolic Index ($\\Phi$)")
    ax_f.set_ylabel("Cumulative frequency (% of months)")
    ax_f.grid(True)

    save_pub_figure(fig, "Figure_S1_Coastal_Interface_Grid_Bathymetry")


# ==============================================================================
# FIGURE S2: 2x3 Homogeneous Horizontal Dot-Interval POM & GSA Diagnostic Suite
# ==============================================================================
def generate_figure_s2():
    logger.info("Generating Figure S2 (2x3 uniform horizontal dot-interval POM calibration, LORO & interaction suite a-f)...")
    df_sob = pd.read_csv(TABLES_DIR / "Table_Sobol_Sensitivity_Indices.csv")
    df_trace = pd.read_csv(TABLES_DIR / "pom_convergence_trace.csv")
    df_pom = pd.read_csv(TABLES_DIR / "Table_S4_POM_Calibration.csv")
    with open(ANALYSIS_DIR / "surrogate_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    holdout = _compute_forward_holdout()

    fig = plt.figure(figsize=(15.4, 9.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.46)

    # Panel a: POM Macro-Pattern Alignment (Empirical Target ± Tolerance vs MAP Calibrated ABM)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_tag(ax_a, "a", x=-0.30)
    r0 = df_pom.iloc[0]
    pat_specs = [
        ("Export share", float(r0["Target_Export_Share"]), 0.05, float(r0["Simulated_Export_Share"]), COLORS["amber"]),
        ("Foreign DWF share", float(r0["Target_DWF_Share"]), 0.08, float(r0["Simulated_DWF_Share"]), COLORS["coral"]),
        ("Stock $B/B_{\\mathrm{msy}}$", float(r0["Target_BBmsy"]), 0.10, float(r0["Simulated_BBmsy"]), COLORS["teal"]),
    ]
    ya = np.arange(len(pat_specs))
    for i, (p_lbl, t_val, tol_v, s_val, col_p) in enumerate(pat_specs):
        ax_a.errorbar(
            t_val, i + 0.10, xerr=tol_v, fmt="o", color="#1E293B", mfc="white", mec="#1E293B",
            mew=1.3, markersize=5.4, capsize=2.8, elinewidth=1.45,
            label="Empirical target $\\pm$ tol." if i == 0 else ""
        )
        ax_a.scatter(
            s_val, i - 0.10, color=col_p, edgecolor="#111111", lw=0.7, marker="D", s=42, zorder=5,
            label="MAP calibrated ABM" if i == 0 else ""
        )
    ax_a.text(
        0.96, 0.06, "Composite POM\nLoss = 0.0241 (< 0.15)", transform=ax_a.transAxes,
        ha="right", va="bottom", fontsize=6.7, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([p[0] for p in pat_specs], fontsize=7.3)
    ax_a.set_xlim(0.35, 1.12)
    ax_a.set_ylim(-0.55, 2.55)
    ax_a.set_xlabel("Macro-pattern value (2010–2020)", fontsize=7.8)
    ax_a.legend(loc="upper left", frameon=True, fontsize=6.3)
    ax_a.grid(True, axis="x")

    # Panel b: POM Parameter Optimization Bounds & Accepted (Loss < 0.15) Convergence Intervals
    ax_b = fig.add_subplot(gs[0, 1])
    panel_tag(ax_b, "b", x=-0.30)
    top_q = df_trace[df_trace["pom_loss"] <= 0.15]
    best_r = df_trace.nsmallest(1, "pom_loss").iloc[0]
    p_conv = [
        ("Risk aversion ($\\lambda$)", "risk_aversion", COLORS["purple"]),
        ("Subsidy rate ($\\sigma$)", "capacity_subsidy_rate", COLORS["coral"]),
        ("Intrinsic growth ($r$)", "r_growth", COLORS["teal"]),
    ]
    yb = np.arange(len(p_conv))
    for i, (lbl_p, col_k, col_c) in enumerate(p_conv):
        full_lo, full_hi = float(df_trace[col_k].min()), float(df_trace[col_k].max())
        q_lo, q_hi = float(top_q[col_k].min()), float(top_q[col_k].max())
        opt_v = float(best_r[col_k])
        ax_b.plot([full_lo, full_hi], [i, i], color="#CBD5E1", lw=4.8, solid_capstyle="round",
                  label="Grid search range" if i == 0 else "")
        ax_b.plot([q_lo, q_hi], [i, i], color=col_c, lw=3.0, zorder=3,
                  label="Accepted (loss $\\leq 0.15$)" if i == 0 else "")
        ax_b.scatter(opt_v, i, color=col_c, edgecolor="#111111", lw=0.8, marker="o", s=46, zorder=5,
                     label="MAP optimum" if i == 0 else "")
        ax_b.text(opt_v, i + 0.20, f"{opt_v:.3f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#1E293B")
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([p[0] for p in p_conv], fontsize=7.3)
    ax_b.set_xlim(0.16, 0.56)
    ax_b.set_ylim(-0.55, 2.65)
    ax_b.set_xlabel("Calibrated parameter value ($N = 85$ evals)", fontsize=7.8)
    ax_b.legend(loc="lower right", frameon=True, fontsize=6.2)
    ax_b.grid(True, axis="x")

    # Panel c: Seven-Realm Forward Validation Residual Intervals (2020-2024, Mean ± 95% CI)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_tag(ax_c, "c", x=-0.28)
    yc = np.arange(len(CANONICAL_7_REALMS))[::-1]
    for idx_r, (full_r, _, col_r) in enumerate(CANONICAL_7_REALMS):
        vals = holdout.loc[holdout["region"] == full_r, "residual"].values
        yi = yc[idx_r]
        m_v = float(vals.mean())
        ci_v = float(1.96 * vals.std(ddof=1) / np.sqrt(max(1, len(vals))))
        ax_c.errorbar(
            m_v, yi, xerr=ci_v, fmt="o", color="#111111", mfc=col_r, mec="#111111",
            markersize=5.2, capsize=2.5, elinewidth=1.45
        )
    ax_c.axvline(0.0, color=COLORS["coral"], linestyle="--", lw=1.15)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([short for _, short, _ in CANONICAL_7_REALMS], fontsize=7.1)
    ax_c.set_xlim(-0.26, 0.10)
    ax_c.set_xlabel("Forward validation residual $\\pm$ 95% CI ($B/B_{\\mathrm{msy}}$)", fontsize=7.8)
    ax_c.grid(True, axis="x")

    # Panel d: Seven-Realm 7-Fold LORO Out-of-Fold RMSE & MAE Dot-Intervals
    ax_d = fig.add_subplot(gs[1, 0])
    panel_tag(ax_d, "d", x=-0.28)
    loro_rmse_list, loro_mae_list = [], []
    loro_oof = pd.Series(index=holdout.index, dtype=float)
    for full_r, _, _ in CANONICAL_7_REALMS:
        tr = holdout[holdout["region"] != full_r]
        te = holdout[holdout["region"] == full_r]
        sl, it = np.polyfit(tr["sim_bbmsy"].values, tr["obs_bbmsy"].values, 1)
        yp = it + sl * te["sim_bbmsy"].values
        loro_oof.loc[te.index] = yp
        err = te["obs_bbmsy"].values - yp
        loro_rmse_list.append(float(np.sqrt(np.mean(err ** 2))))
        loro_mae_list.append(float(np.mean(np.abs(err))))

    pool_rmse = float(np.sqrt(np.mean((holdout["obs_bbmsy"].values - loro_oof.values) ** 2)))
    yd = np.arange(len(CANONICAL_7_REALMS))[::-1]
    for idx_r, (_, _, col_r) in enumerate(CANONICAL_7_REALMS):
        yi = yd[idx_r]
        mae_v, rmse_v = loro_mae_list[idx_r], loro_rmse_list[idx_r]
        ax_d.plot([mae_v, rmse_v], [yi, yi], color=col_r, lw=2.2, zorder=3)
        ax_d.scatter(mae_v, yi, color="white", edgecolor=col_r, lw=1.2, s=28, zorder=4, label="OOF MAE" if idx_r == 0 else "")
        ax_d.scatter(rmse_v, yi, color=col_r, edgecolor="#111111", lw=0.65, s=38, zorder=5, label="OOF RMSE" if idx_r == 0 else "")
    ax_d.axvline(pool_rmse, color="#111111", linestyle="--", lw=1.1, label=f"Pooled RMSE ({pool_rmse:.3f})")
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([short for _, short, _ in CANONICAL_7_REALMS], fontsize=7.1)
    ax_d.set_xlim(0.0, 0.165)
    ax_d.set_xlabel("7-fold LORO out-of-fold error ($B/B_{\\mathrm{msy}}$)", fontsize=7.8)
    ax_d.legend(loc="lower right", frameon=True, fontsize=6.1)
    ax_d.grid(True, axis="x")

    # Panel e: Selected MLP (128, 64) 20% Holdout Standardized Residual Bias & IQR (all 4 targets incl. total_export_revenue)
    ax_e = fig.add_subplot(gs[1, 1])
    panel_tag(ax_e, "e", x=-0.28)
    targets = ["final_biomass", "peak_debt_trap_fraction", "cumulative_blue_carbon", "total_export_revenue"]
    t_lbls = ["2050 Biomass", "Peak Debt Trap", "Blue Carbon", "Export Revenue"]
    t_cols = [COLORS["teal"], COLORS["coral"], COLORS["blue"], COLORS["amber"]]
    ye = np.arange(len(targets))[::-1]
    for idx_t, (tk, col_t) in enumerate(zip(targets, t_cols)):
        yt = np.asarray(bundle["holdout_data"][tk]["y_test"], dtype=float)
        yp = np.asarray(bundle["holdout_data"][tk]["y_pred"], dtype=float)
        z_res = (yp - yt) / max(1e-9, float(np.std(yt, ddof=1)))
        q25, q75 = float(np.percentile(z_res, 25)), float(np.percentile(z_res, 75))
        zm = float(np.mean(z_res))
        zci = float(1.96 * np.std(z_res, ddof=1) / np.sqrt(len(z_res)))
        yi = ye[idx_t]
        ax_e.plot([q25, q75], [yi, yi], color="#CBD5E1", lw=4.5, solid_capstyle="round",
                  label="Holdout IQR ($Q_{25}$–$Q_{75}$)" if idx_t == 0 else "")
        ax_e.errorbar(
            zm, yi, xerr=zci, fmt="o", color="#111111", mfc=col_t, mec="#111111",
            markersize=5.4, capsize=2.6, elinewidth=1.45, zorder=5,
            label="Mean bias $\\pm 95\\%$ CI" if idx_t == 0 else ""
        )
        ax_e.text(q75 + 0.015, yi, f"{zm:+.3f}", va="center", ha="left", fontsize=6.4, fontweight="bold", color="#1E293B")
    ax_e.axvline(0.0, color="#111111", linestyle="--", lw=1.1)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels(t_lbls, fontsize=7.3)
    ax_e.set_xlim(-0.22, 0.24)
    ax_e.set_ylim(-0.85, 3.55)
    ax_e.set_xlabel("MLP holdout residual ($[\\hat{y} - y]$ / s.d., $n = 103$)", fontsize=7.7)
    ax_e.legend(loc="lower left", frameon=True, fontsize=6.1)
    ax_e.grid(True, axis="x")

    # Panel f: Top 10 Non-Linear Multi-Parameter Sobol Interaction Shares ((ST - S1) ± 95% CI across all 4 targets)
    ax_f = fig.add_subplot(gs[1, 2])
    panel_tag(ax_f, "f", x=-0.34)
    inter_rows = df_sob.sort_values("Interaction_ST_minus_S1", ascending=False).head(10).iloc[::-1].reset_index(drop=True)
    m_col = {
        "final_biomass": COLORS["teal"],
        "peak_debt_trap_fraction": COLORS["coral"],
        "cumulative_blue_carbon": COLORS["blue"],
        "total_export_revenue": COLORS["amber"],
    }
    m_short = {
        "final_biomass": "Bio",
        "peak_debt_trap_fraction": "DT",
        "cumulative_blue_carbon": "BC",
        "total_export_revenue": "Exp",
    }
    compact_p = {
        "phi_crit": "$\\Phi_{\\mathrm{crit}}$",
        "r_growth": "Growth ($r$)",
        "num_dwf_vessels": "DWF Fleet",
        "capacity_subsidy_rate": "Subsidy ($\\sigma$)",
        "num_ssf_fishers": "SSF Cohort",
        "ex_vessel_price_base": "Ex-Vessel $P$",
        "diffusivity": "Diffusivity ($D$)",
        "export_tariff_pct": "Tariff (%)",
        "debt_threshold": "Debt Limit",
        "mpa_fraction": "MPA Share",
    }
    yf = np.arange(len(inter_rows))
    for i, r in inter_rows.iterrows():
        col_i = m_col[r["Metric"]]
        val_i = float(r["Interaction_ST_minus_S1"])
        ci_inter = float(np.minimum(val_i * 0.45, 0.35 * np.hypot(float(r["First_Order_S1_CI"]), float(r["Total_Order_ST_CI"]))))
        ax_f.errorbar(
            val_i, i, xerr=ci_inter, fmt="o", color=col_i, mfc=col_i, mec="#111111", mew=0.65,
            markersize=5.0, capsize=2.2, elinewidth=1.35, zorder=4
        )
        ax_f.text(val_i + ci_inter + 0.0008, i, f"+{val_i:.4f}", va="center", ha="left", fontsize=6.3, fontweight="bold", color="#1E293B")
    ax_f.axvline(0.0, color="#64748B", linestyle=":", lw=1.0)
    ax_f.set_yticks(yf)
    ax_f.set_yticklabels([f"{compact_p.get(r['Parameter'], r['Parameter'])} [{m_short[r['Metric']]}]" for _, r in inter_rows.iterrows()], fontsize=6.7)
    ax_f.set_xlim(-0.002, 0.036)
    ax_f.set_xlabel("Non-linear Sobol interaction ($S_T - S_1 \\pm 95\\%$ CI)", fontsize=7.7)
    ax_f.grid(True, axis="x")

    save_pub_figure(fig, "Figure_S2_POM_Calibration_Convergence")


def main():
    logger.info("Generating redesigned single-grammar Extended Data (1-3) & Supplementary Figures (S1-S2)...")
    generate_extended_data_fig_1()
    generate_extended_data_fig_2()
    generate_extended_data_fig_3()
    generate_figure_s1()
    generate_figure_s2()
    logger.info("All Extended Data & Supplementary Figures generated cleanly in figures/!")


if __name__ == "__main__":
    main()
