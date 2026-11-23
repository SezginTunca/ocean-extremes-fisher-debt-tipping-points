#!/usr/bin/env python3
"""
style_and_maps.py
Unified Nature Climate Change visual styling, locked colorblind-safe semantic palettes,
Equal-Earth (EPSG:8857) global cartography, co-registered 0.1-degree coastal shelf grid
renderers, and reusable single-grammar raincloud/interval helpers for ABM-Earth-TropEcS.

Follows publication visualization standards:
- Strict within-figure single visual grammar (zero chaotic chart-type mixing)
- Arial / Helvetica typography (8–11.5 pt legible at print column widths)
- Zero cluttered internal subplot titles (clean bold lowercase panel tags + subtle metric badges)
- Colorblind-safe Okabe-Ito & sequential/diverging scientific palettes
- 600 DPI PNG + vector PDF dual export
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
GEOJSON_PATH = DATA_DIR / "geo" / "ne_50m_coastal_countries.geojson"

# Locked colorblind-safe Okabe-Ito & publication palette
COLORS = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "teal": "#009E73",
    "coral": "#D55E00",
    "amber": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6C7A89",
    "light_gray": "#E5E7EB",
    "ocean": "#EEF4F8",
    "light_teal": "#E6F4F1",
    "light_coral": "#FBEAE7",
    "dark": "#1A1D24",
}

SSP_COLORS = {
    "Historical": "#222222",
    "SSP1-2.6": "#0072B2",
    "SSP2-4.5": "#E69F00",
    "SSP5-8.5": "#D55E00",
    "Recovery": "#009E73",
}

POLICY_COLORS = {
    "Status Quo": "#D55E00",
    "30x30 Spatial Closures": "#E69F00",
    "WTO Subsidy Ban": "#0072B2",
    "Blue Carbon Dividend": "#CC79A7",
    "Integrated Policy Package": "#009E73",
}

POLICY_SHORT = {
    "Status Quo": "Status Quo",
    "WTO Subsidy Ban": "WTO Subsidy Ban",
    "30x30 Spatial Closures": "30×30 MPA",
    "Blue Carbon Dividend": "Blue-Carbon Dividend",
    "Integrated Policy Package": "Integrated Package",
}

SOLVENCY_COLORS = {
    "Solvent": "#009E73",
    "Trapped": "#D55E00",
    "Exited": "#6C7A89",
}

# Canonical 7 maritime realms locked across EVERY global/regional chart
CANONICAL_7_REALMS: List[Tuple[str, str, str]] = [
    ("West Africa Upwelling", "West Africa", "#D55E00"),
    ("Coral Triangle / SE Asia", "Coral Triangle", "#009E73"),
    ("Western Indian Ocean", "W. Indian Ocean", "#CC79A7"),
    ("Humboldt / Eastern Pacific", "Humboldt", "#E69F00"),
    ("Mediterranean & Black Sea", "Mediterranean", "#56B4E9"),
    ("Other Coastal Nations", "Other Coastal", "#7F8C8D"),
    ("Distant-Water & Processing Hubs", "DWF & Hub States", "#0072B2"),
]

REALM_SHORT_MAP = {full: short for full, short, _ in CANONICAL_7_REALMS}
REALM_COLOR_MAP = {full: col for full, _, col in CANONICAL_7_REALMS}
REALM_SHORT_COLOR_MAP = {short: col for _, short, col in CANONICAL_7_REALMS}

TARGET_METRIC_LABELS = {
    "final_biomass": "2050 Biomass (t)",
    "peak_debt_trap_fraction": "Peak Debt Trap",
    "cumulative_blue_carbon": "Blue Carbon ($\\mathrm{tCO_2e}$)",
    "total_export_revenue": "Export Revenue (US$)",
}

PARAM_LABELS = {
    "r_growth": "Intrinsic Growth ($r$)",
    "phi_crit": "Critical Metabolic ($\\Phi_{\\mathrm{crit}}$)",
    "capacity_subsidy_rate": "Subsidy Rate ($\\sigma$)",
    "fuel_price_volatility": "Fuel Volatility",
    "risk_aversion": "Risk Aversion ($\\lambda$)",
    "debt_threshold": "Debt Limit ($D_{\\mathrm{thresh}}$)",
    "carbon_price": "Carbon Price ($P_C$)",
    "export_tariff_pct": "Export Tariff (%)",
    "mpa_fraction": "MPA Share (30×30)",
    "mangrove_fraction": "Mangrove Share ($\\mu_0$)",
    "carrying_capacity_k": "Carrying Capacity ($K_0$)",
    "num_ssf_fishers": "SSF Cohort ($N_{\\mathrm{ssf}}$)",
    "num_dwf_vessels": "DWF Fleet ($N_{\\mathrm{dwf}}$)",
    "sst_warming_rate": "SST Warming Rate",
    "base_fuel_price": "Baseline Fuel Price",
    "ex_vessel_price_base": "Ex-Vessel Fish Price",
    "diffusivity": "Spatial Diffusivity ($D$)",
}


def apply_nature_style():
    """Enforces print-true Nature Climate Change Arial/Helvetica typography and spine hygiene."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.default": "regular",
        "font.size": 8.6,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.2,
        "legend.title_fontsize": 7.4,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.68,
        "ytick.major.width": 0.68,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelcolor": "#111111",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_tag(ax, letter: str, subtitle: str = "", x: float = -0.11, y: float = 1.04, despine: bool = True):
    """
    Places clean bold lowercase Nature panel letter outside top-left spine.
    Omits cluttered bold subplot titles inside figures for publication clarity.
    """
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=11.2, fontweight="bold", va="bottom", ha="left", color="#000000"
    )


def save_pub_figure(fig, name: str):
    """Exports figure in 600 DPI PNG and vector PDF inside figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


@lru_cache(maxsize=1)
def load_coastal_geojson() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Returns (lonlat_gdf, equal_earth_gdf) from self-contained repo GeoJSON."""
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Missing self-contained GeoJSON at {GEOJSON_PATH}")
    gdf_ll = gpd.read_file(GEOJSON_PATH)
    gdf_eq = gdf_ll.to_crs("EPSG:8857")
    return gdf_ll, gdf_eq


_CORRIDOR_EEZ_MAP = {
    "Southwest Atlantic": ["ARG", "URY", "BRA"],
    "Offshore Mauritania / Senegal Boundary": ["MRT", "SEN", "GMB", "GIN", "GNB"],
    "Western Pacific High Seas Pocket": ["KIR", "FSM", "PNG", "PHL", "IDN"],
    "Offshore Namibia": ["NAM", "ZAF"],
    "Offshore Peru": ["PER", "CHL"],
    "Offshore Angola": ["AGO", "COG", "COD", "GAB", "NGA", "GHA", "CIV", "SLE", "LBR", "CMR"],
    "Solomon / PNG High Seas Enclave": ["SLB", "PNG", "VUT", "FJI", "WSM"],
    "Western Indian Ocean High Seas Pocket": ["TZA", "KEN", "MOZ", "MUS", "COM", "LKA", "MDV", "YEM", "OMN"],
    "Seychelles / Madagascar Channel Corridor": ["SYC", "MDG", "MOZ", "TZA", "COM", "MUS"],
    "Galapagos EEZ Perimeter / Equatorial Pacific": ["ECU", "COL", "PAN", "CRI", "MEX"],
    "Offshore Western Sahara / Morocco Upwelling": ["MAR", "MRT", "SEN"],
}

# National mangrove extent (km²) from Global Mangrove Watch v3.0 (Bunting et al. 2022) & Bertram et al. (2021);
# strictly 0.0 km² for temperate, polar, Mediterranean, and hyper-arid non-mangrove coastlines.
GMW_NATIONAL_MANGROVE_KM2: Dict[str, float] = {
    "IDN": 29500.0, "BRA": 11100.0, "AUS": 9910.0, "MEX": 7750.0, "NGA": 6900.0,
    "MYS": 5200.0, "MMR": 4950.0, "PNG": 4600.0, "BGD": 4400.0, "CUB": 4200.0,
    "IND": 4100.0, "COL": 2800.0, "MOZ": 2700.0, "MDG": 2650.0, "PHL": 2600.0,
    "THA": 2450.0, "VNM": 2150.0, "GNB": 2100.0, "GIN": 2000.0, "CMR": 1950.0,
    "USA": 1900.0, "GAB": 1700.0, "ECU": 1550.0, "PAN": 1500.0, "SEN": 1280.0,
    "SLE": 1250.0, "TZA": 1150.0, "PAK": 950.0, "GMB": 580.0, "KEN": 540.0,
    "SLB": 530.0, "FJI": 490.0, "KHM": 480.0, "CRI": 390.0, "AGO": 320.0,
    "CHN": 280.0, "NZL": 260.0, "DOM": 210.0, "LKA": 190.0, "GHA": 140.0,
    "LBR": 110.0, "IRN": 105.0, "CIV": 95.0, "FSM": 88.0, "SAU": 85.0,
    "ARE": 75.0, "PER": 55.0, "COG": 35.0, "COD": 30.0, "ZAF": 30.0,
    "VUT": 25.0, "YEM": 22.0, "OMN": 12.0, "JPN": 10.0, "QAT": 8.0,
    "EGY": 6.0, "WSM": 4.0, "SYC": 3.5, "KIR": 3.0, "COM": 1.5,
    "MUS": 1.5, "BHR": 1.0, "MDV": 1.0, "KWT": 0.5,
}


def compute_coastal_corridor_loitering_panel(start_year: int = 2012, end_year: int = 2024) -> pd.DataFrame:
    """
    Computes annual (start_year..end_year) unreported offshore transshipment and untracked
    industrial vessel hours along coastal EEZ perimeters across all 109 sovereign coastal states
    without open-registry flag-of-convenience carrier attribution.
    """
    pan = pd.read_parquet(DATA_DIR / "empirical_panel.parquet")
    base = pan[(pan["year"] >= start_year) & (pan["year"] <= end_year)][
        ["country_iso3", "year", "region_classification"]
    ].drop_duplicates(["country_iso3", "year"]).copy()
    valid_isos = set(base["country_iso3"].unique())

    events_path = RAW_DIR / "gfw_carrier_loitering_events.csv"
    ocean_path = RAW_DIR / "ocean_economy_panel_109.csv"

    rows = []
    if events_path.exists():
        ev = pd.read_csv(events_path)
        ev = ev[(ev["year"] >= start_year) & (ev["year"] <= end_year)]
        for _, r in ev.iterrows():
            isos = _CORRIDOR_EEZ_MAP.get(str(r.get("nearest_eez", "")), [])
            n_iso = max(len(isos), 1)
            for iso in isos:
                if iso in valid_isos:
                    rows.append({
                        "country_iso3": iso,
                        "year": int(r["year"]),
                        "corridor_hrs": float(r.get("duration_hours", 0.0)) / n_iso,
                    })
    if rows:
        df_ev = pd.DataFrame(rows).groupby(["country_iso3", "year"], as_index=False)["corridor_hrs"].sum()
        base = base.merge(df_ev, on=["country_iso3", "year"], how="left")
        base["corridor_hrs"] = base["corridor_hrs"].fillna(0.0)
    else:
        base["corridor_hrs"] = 0.0

    if ocean_path.exists():
        oc = pd.read_csv(ocean_path)
        oc = oc[(oc["year"] >= start_year) & (oc["year"] <= end_year)]
        if "gfw_dark_vessel_hours" in oc.columns:
            oc_sub = oc[["country_iso3", "year", "gfw_dark_vessel_hours"]].drop_duplicates(["country_iso3", "year"])
            base = base.merge(oc_sub, on=["country_iso3", "year"], how="left")
            base["corridor_hrs"] = base["corridor_hrs"] + 0.15 * base["gfw_dark_vessel_hours"].fillna(0.0)
            base.drop(columns=["gfw_dark_vessel_hours"], inplace=True)

    return base


def compute_coastal_corridor_loitering() -> Dict[str, float]:
    """
    Computes 2015-2023 annual mean unreported offshore transshipment and untracked vessel hours
    without open-registry flag-of-convenience carrier attribution (eliminating PAN/LBR/VUT/CYP distortion).
    """
    df_2015_2023 = compute_coastal_corridor_loitering_panel(start_year=2015, end_year=2023)
    return {
        iso: float(val)
        for iso, val in df_2015_2023.groupby("country_iso3")["corridor_hrs"].mean().items()
    }


def compute_mangrove_blue_carbon_suitability() -> pd.DataFrame:
    """
    Constructs an empirical Global Mangrove Watch v3.0 (Bunting et al. 2022) & Bertram et al. (2021)
    mangrove & coastal blue-carbon index [0, 1] across all 109 sovereign coastal states.
    """
    prof = pd.read_csv(DATA_DIR / "country_profiles.csv")
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    geo_df = pd.DataFrame([f["properties"] for f in gj["features"]]).rename(columns={"ADM0_A3": "country_iso3"})
    out = prof.merge(geo_df[["country_iso3", "rep_lat", "rep_lon", "area_km2"]], on="country_iso3", how="left")

    gmw_km2 = out["country_iso3"].map(lambda c: GMW_NATIONAL_MANGROVE_KM2.get(c, 0.0)).astype(float)
    out["gmw_mangrove_km2"] = gmw_km2
    iwp_carbon_density = np.where(
        out["region_classification"].isin(["Coral Triangle / SE Asia", "Western Indian Ocean"]),
        1.25, 1.0
    )
    mang_extent_norm = np.log1p(gmw_km2 * iwp_carbon_density) / np.log1p(29500.0 * 1.25)
    mang_density_norm = np.clip(np.sqrt(gmw_km2 / np.maximum(out["area_km2"], 50.0)) * 3.0, 0.0, 1.0)
    cwon_authentic = {"AUS": 10.588749, "USA": 3.507772, "JPN": 1.681937, "CHN": 1.217758}
    cwon_bonus = out["country_iso3"].map(lambda c: min(0.12, cwon_authentic.get(c, 0.0) / 90.0)).astype(float)

    out["mangrove_bc_index"] = np.clip(
        (0.68 * mang_extent_norm + 0.27 * mang_density_norm) * iwp_carbon_density + cwon_bonus,
        0.0, 1.0
    )
    return out


@lru_cache(maxsize=1)
def _ocean_and_graticule_eq():
    """Precomputes Equal Earth ocean oval outline and 30-deg graticule polylines."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:8857", always_xy=True)

    lons_top = np.linspace(-179.9, 179.9, 120)
    lats_right = np.linspace(83.6, -62.0, 80)
    lons_bot = np.linspace(179.9, -179.9, 120)
    lats_left = np.linspace(-62.0, 83.6, 80)

    ring_lon = np.concatenate([lons_top, np.full_like(lats_right, 179.9), lons_bot, np.full_like(lats_left, -179.9)])
    ring_lat = np.concatenate([np.full_like(lons_top, 83.6), lats_right, np.full_like(lons_bot, -62.0), lats_left])
    ox, oy = transformer.transform(ring_lon, ring_lat)
    ocean_xy = np.column_stack([ox, oy])

    grat_lines = []
    for lon in range(-150, 180, 30):
        lats = np.linspace(-62.0, 83.6, 80)
        gx, gy = transformer.transform(np.full_like(lats, float(lon)), lats)
        grat_lines.append((gx, gy))
    for lat in [-45, -30, 0, 30, 60]:
        lons = np.linspace(-179.9, 179.9, 120)
        gx, gy = transformer.transform(lons, np.full_like(lons, float(lat)))
        grat_lines.append((gx, gy))
    return ocean_xy, grat_lines, transformer


def plot_global_equal_earth(
    ax,
    values_by_iso3: Optional[Dict[str, float]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[mcolors.Normalize] = None,
    cbar_label: str = "",
    categorical_colors: Optional[Dict[str, str]] = None,
    bubble_by_iso3: Optional[Dict[str, float]] = None,
    bubble_scale: float = 0.08,
    bubble_color: str = "#D55E00",
    direct_ram_isos: Optional[set] = None,
    region_boxes: Optional[List[Tuple[float, float, float, float, str, str]]] = None,
    cbar_orientation: str = "bottom",
    cbar_pad: float = 0.045,
    summary_badge: str = "",
):
    """
    Renders a co-registered Equal-Earth (EPSG:8857) choropleth map of the
    109 sovereign coastal states with centroid markers for micro-island nations
    and uniform horizontal bottom colorbars.
    """
    _, gdf_eq = load_coastal_geojson()
    ocean_xy, grat_lines, transformer = _ocean_and_graticule_eq()

    ax.add_patch(MplPolygon(ocean_xy, closed=True, facecolor=COLORS["ocean"], edgecolor="#94A3B8", linewidth=0.75, zorder=0))
    for gx, gy in grat_lines:
        ax.plot(gx, gy, color="#D5DFEB", linewidth=0.36, zorder=1)

    study_isos = set(values_by_iso3.keys()) if values_by_iso3 is not None else set(categorical_colors.keys())
    bg = gdf_eq[~gdf_eq["ADM0_A3"].isin(study_isos)]
    bg.plot(ax=ax, color=COLORS["light_gray"], edgecolor="#FFFFFF", linewidth=0.25, zorder=2)

    study = gdf_eq[gdf_eq["ADM0_A3"].isin(study_isos)].copy()
    if categorical_colors is not None:
        study["_color"] = study["ADM0_A3"].map(categorical_colors)
        study.plot(ax=ax, color=study["_color"], edgecolor="#2B303A", linewidth=0.32, zorder=3)
        sm = None
    else:
        study["_val"] = study["ADM0_A3"].map(values_by_iso3).astype(float)
        if norm is None:
            if vmin is None:
                vmin = float(study["_val"].min())
            if vmax is None:
                vmax = float(study["_val"].max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        study["_color"] = [mcolors.to_hex(cmap_obj(norm(v))) for v in study["_val"]]
        study.plot(ax=ax, color=study["_color"], edgecolor="#2B303A", linewidth=0.32, zorder=3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])

    micro = study[study["area_km2"] < 25000].copy()
    if not micro.empty:
        mx, my = transformer.transform(micro["rep_lon"].values, micro["rep_lat"].values)
        ax.scatter(
            mx, my, c=micro["_color"].tolist(), s=15, marker="o",
            edgecolors="#111111", linewidths=0.45, zorder=4
        )

    if direct_ram_isos:
        ram_sub = study[study["ADM0_A3"].isin(direct_ram_isos)]
        rx, ry = transformer.transform(ram_sub["rep_lon"].values, ram_sub["rep_lat"].values)
        ax.scatter(rx, ry, s=24, marker="D", facecolors="none", edgecolors="#111111", linewidths=0.95, zorder=5)

    if bubble_by_iso3:
        b_rows = study[study["ADM0_A3"].isin(bubble_by_iso3.keys())].copy()
        b_vals = np.array([max(0.0, float(bubble_by_iso3[iso])) for iso in b_rows["ADM0_A3"]])
        bx, by = transformer.transform(b_rows["rep_lon"].values, b_rows["rep_lat"].values)
        mask_pos = b_vals > 0.5
        ax.scatter(
            bx[mask_pos], by[mask_pos],
            s=np.sqrt(b_vals[mask_pos]) * bubble_scale * 42.0 + 4.0,
            color=bubble_color, alpha=0.58, edgecolors="#111111", linewidths=0.55, zorder=6
        )

    if region_boxes:
        for (x0, x1, y0, y1, col_box, box_lbl) in region_boxes:
            lons_b = np.concatenate([np.linspace(x0, x1, 20), np.full(20, x1), np.linspace(x1, x0, 20), np.full(20, x0)])
            lats_b = np.concatenate([np.full(20, y0), np.linspace(y0, y1, 20), np.full(20, y1), np.linspace(y1, y0, 20)])
            px, py = transformer.transform(lons_b, lats_b)
            ax.plot(px, py, color=col_box, linewidth=1.2, linestyle="-", zorder=7)
            if box_lbl:
                tx, ty = transformer.transform(x0 + 1.5, y1 + 2.0)
                ax.text(tx, ty, box_lbl, fontsize=6.4, fontweight="bold", color=col_box, zorder=8)

    if summary_badge:
        ax.text(
            0.50, 1.015, summary_badge, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.4, fontweight="semibold", color="#1E293B"
        )

    ax.set_xlim(ocean_xy[:, 0].min() * 1.01, ocean_xy[:, 0].max() * 1.01)
    ax.set_ylim(ocean_xy[:, 1].min() * 1.02, ocean_xy[:, 1].max() * 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    if sm is not None and cbar_label:
        div = make_axes_locatable(ax)
        if cbar_orientation == "bottom":
            cax = div.append_axes("bottom", size="5.8%", pad=cbar_pad)
            cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
        else:
            cax = div.append_axes("right", size="3.2%", pad=cbar_pad)
            cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, fontsize=7.7)
        cbar.ax.tick_params(labelsize=7.0)
        return cbar
    return None


def plot_regional_submap(
    ax,
    lon_bounds: Tuple[float, float],
    lat_bounds: Tuple[float, float],
    values_by_iso3: Optional[Dict[str, float]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    categorical_colors: Optional[Dict[str, str]] = None,
    badge_label: str = "",
    bubble_by_iso3: Optional[Dict[str, float]] = None,
    bubble_color: str = "#D55E00",
    bubble_scale: float = 0.08,
    annotate_isos: Optional[List[str]] = None,
    cmip_grid_deg: float = 0.0,
    shelf_box_xy: Optional[Tuple[float, float]] = None,
):
    """Focused regional coastal zoom submap helper."""
    gdf_ll, _ = load_coastal_geojson()
    ax.set_facecolor(COLORS["ocean"])

    study_isos = set(values_by_iso3.keys()) if values_by_iso3 is not None else set(categorical_colors.keys())
    bg = gdf_ll[~gdf_ll["ADM0_A3"].isin(study_isos)]
    bg.plot(ax=ax, color=COLORS["light_gray"], edgecolor="#CBD5E1", linewidth=0.35, zorder=1)

    study = gdf_ll[gdf_ll["ADM0_A3"].isin(study_isos)].copy()
    if categorical_colors is not None:
        study["_color"] = study["ADM0_A3"].map(categorical_colors)
        study.plot(ax=ax, color=study["_color"], edgecolor="#222222", linewidth=0.45, zorder=2)
    else:
        study["_val"] = study["ADM0_A3"].map(values_by_iso3).astype(float)
        if vmin is None:
            vmin = float(study["_val"].min())
        if vmax is None:
            vmax = float(study["_val"].max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        study["_color"] = [mcolors.to_hex(cmap_obj(norm(v))) for v in study["_val"]]
        study.plot(ax=ax, color=study["_color"], edgecolor="#222222", linewidth=0.45, zorder=2)

    if badge_label:
        ax.text(
            0.03, 0.95, badge_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6, fontweight="bold", color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.90), zorder=8
        )

    ax.set_xlim(lon_bounds)
    ax.set_ylim(lat_bounds)
    mid_lat = 0.5 * (lat_bounds[0] + lat_bounds[1])
    ax.set_aspect(1.0 / max(0.45, float(np.cos(np.deg2rad(mid_lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#475569")
        sp.set_linewidth(0.7)


def plot_shelf_submap(
    ax,
    grid2d: np.ndarray,
    cmap: str = "viridis",
    cbar_label: str = "",
    mpa_mask: Optional[np.ndarray] = None,
    mpa_edge_color: str = "#FFFFFF",
    port_xy: Optional[Tuple[float, float]] = None,
    contour_val: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[mcolors.Normalize] = None,
    cbar_pad: float = 0.05,
):
    """Co-registered 10x10 (0.1 deg / 100 km) cross-shelf submap with unified coordinates and high-contrast MPA borders."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#222222")
        sp.set_linewidth(0.68)

    if norm is not None:
        im = ax.imshow(grid2d, cmap=cmap, origin="lower", norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(grid2d, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5.2%", pad=cbar_pad)
    cbar = plt.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.9)

    if mpa_mask is not None:
        h, w = mpa_mask.shape
        for r in range(h):
            for c in range(w):
                if mpa_mask[r, c]:
                    ax.add_patch(
                        Rectangle(
                            (c - 0.44, r - 0.44), 0.88, 0.88,
                            fill=False, edgecolor="#111111", linewidth=1.65, linestyle="-", zorder=4
                        )
                    )
                    ax.add_patch(
                        Rectangle(
                            (c - 0.44, r - 0.44), 0.88, 0.88,
                            fill=False, edgecolor=mpa_edge_color, linewidth=1.05, linestyle="--", zorder=5
                        )
                    )
    if port_xy is not None:
        ax.scatter([port_xy[0]], [port_xy[1]], color=COLORS["coral"], edgecolors="#FFFFFF", linewidths=0.85, s=95, marker="*", zorder=7)
    if contour_val is not None and float(grid2d.min()) < contour_val < float(grid2d.max()):
        cs = ax.contour(grid2d, levels=[contour_val], colors=["#111111"], linewidths=2.0, linestyles="-", zorder=6)
        lbls = ax.clabel(cs, inline=True, fontsize=6.9, fmt=f"$\\Phi$={contour_val:.1f}")
        for txt_obj in lbls:
            txt_obj.set_fontweight("bold")
            txt_obj.set_bbox(dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#111111", lw=0.4, alpha=0.92))
            txt_obj.set_zorder(8)

    ax.set_xticks([0, 4.5, 9])
    ax.set_xticklabels(["0", "50", "100 km"], fontsize=7.2)
    ax.set_yticks([0, 4.5, 9])
    ax.set_yticklabels(["0", "50", "100"], fontsize=7.2)
    return im


def plot_raincloud_h(
    ax,
    groups: Sequence[Tuple[str, np.ndarray, str, str]],
    xlabel: str,
    ref_line: Optional[float] = None,
    ref_label: str = "",
    shade_band: Optional[Tuple[float, float]] = None,
    seed: int = 42,
    xlim_pad_right: float = 0.22,
):
    """
    Renders a clean, single-grammar horizontal raincloud-interval chart:
    groups = [(y_label, values_1d, hex_color, right_annotation_text), ...]
    Each row displays:
      - Smooth upper half-violin density above y + 0.10
      - Jittered individual sample dots below y - 0.14
      - Mean ± 95% CI horizontal interval at y + 0.10
      - Crisp right-side summary value annotation without spine collisions
    """
    rng = np.random.default_rng(seed)
    n_g = len(groups)
    all_min, all_max = np.inf, -np.inf

    if shade_band is not None:
        ax.axvspan(shade_band[0], shade_band[1], color="#CBD5E1", alpha=0.28, zorder=1)
        all_min = min(all_min, float(shade_band[0]))
        all_max = max(all_max, float(shade_band[1]))

    for idx, (_, vals, col, _) in enumerate(groups):
        arr = np.asarray(vals, dtype=float)
        all_min = min(all_min, float(arr.min()))
        all_max = max(all_max, float(arr.max()))

        if np.std(arr) > 1e-8 and len(arr) >= 3:
            parts = ax.violinplot(
                [arr], positions=[idx + 0.10], vert=False, widths=0.52,
                showmeans=False, showextrema=False, bw_method=0.38
            )
            for pc in parts["bodies"]:
                verts = pc.get_paths()[0].vertices
                verts[:, 1] = np.clip(verts[:, 1], idx + 0.10, np.inf)
                pc.set_facecolor(col)
                pc.set_edgecolor("#222222")
                pc.set_linewidth(0.65)
                pc.set_alpha(0.48)

        jit = rng.uniform(-0.08, 0.08, size=len(arr))
        ax.scatter(
            arr, np.full_like(arr, idx - 0.14) + jit,
            color=col, s=13, alpha=0.68, edgecolors="#222222", linewidths=0.28, zorder=3
        )

        m_v = float(np.mean(arr))
        ci_v = float(1.96 * np.std(arr, ddof=1) / np.sqrt(max(1, len(arr)))) if len(arr) > 1 else 0.0
        ax.errorbar(
            m_v, idx + 0.10, xerr=ci_v, fmt="o",
            color="#111111", mfc=col, mec="#111111",
            markersize=5.2, capsize=2.6, elinewidth=1.45, zorder=6
        )

    if ref_line is not None:
        all_min = min(all_min, ref_line)
        all_max = max(all_max, ref_line)
        ax.axvline(ref_line, color="#1E293B", linestyle="--", lw=1.15, zorder=2)
        if ref_label:
            ax.text(
                ref_line, n_g - 0.35, f" {ref_label} ",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
                zorder=7
            )

    span = max(1e-6, all_max - all_min)
    x_lo = all_min - 0.08 * span
    x_hi = all_max + xlim_pad_right * span
    ax.set_xlim(x_lo, x_hi)

    for idx, (_, vals, _, ann_txt) in enumerate(groups):
        if ann_txt:
            arr = np.asarray(vals, dtype=float)
            m_v = float(np.mean(arr))
            ci_v = float(1.96 * np.std(arr, ddof=1) / np.sqrt(max(1, len(arr)))) if len(arr) > 1 else 0.0
            x_anchor = max(float(arr.max()), m_v + ci_v) + 0.028 * span
            ax.text(
                x_anchor, idx + 0.10, ann_txt,
                ha="left", va="center", fontsize=6.7, fontweight="bold", color="#1E293B", zorder=7
            )

    ax.set_yticks(np.arange(n_g))
    ax.set_yticklabels([g[0] for g in groups], fontsize=7.3)
    ax.set_ylim(-0.48, n_g - 0.25)
    ax.set_xlabel(xlabel, fontsize=8.1)
    ax.grid(True, axis="x")
