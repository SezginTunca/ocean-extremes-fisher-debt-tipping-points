# Ocean extremes and fisher debt drive tipping points in coastal blue economies

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![ODD+D Standard](https://img.shields.io/badge/standard-ODD%2BD-success.svg)](model/odd_protocol.md)

## Overview

This repository provides exclusively the **15 core scientific Python modules** and formal ODD+D protocol (`ABM-Earth-TropEcS`) supporting the research article:

> **Tunca, S.** (2026). *Ocean extremes and fisher debt drive tipping points in coastal blue economies*. Manuscript submitted to *Nature Climate Change*.  
> **Author:** Sezgin Tunca (`sezgin.tunca@gmail.com`), Department of Business and Sustainability, University of Southern Denmark ([ORCID: 0000-0001-6840-1466](https://orcid.org/0000-0001-6840-1466))  
> **GitHub Repository:** <https://github.com/SezginTunca/ocean-extremes-fisher-debt-tipping-points>

Global climate models project ocean warming and deoxygenation, yet treat human fishing pressure as an exogenous rate and miss shallow continental shelf dynamics alongside rural credit constraints. Coupling process-based CMIP6 shelf projections (`GFDL-ESM4` and `MPI-ESM1-2-HR`) with a five-agent spatial bioeconomic model calibrated across **109 sovereign coastal nations** (`2000–2024`), these modules implement the empirical harmonization, cross-shelf oxygen–metabolic downscaling, safety-first fisher debt dynamics, country-clustered panel regressions, early-warning tipping diagnostics, causal mechanism ablations, neural surrogate Sobol sensitivity analysis, and visualization pipelines reported in the manuscript.

---

## Data-Sharing Policy & Scope

> [!IMPORTANT]
> **Data Distribution Notice:**
> **No raw microdata, third-party tabular extracts, spatial boundary files, or processed empirical datasets are distributed in this GitHub repository.**
>
> In strict respect of third-party public and institutional data-provider terms (Global Fishing Watch, UN Comtrade, FAO FishStatJ, OECD Fisheries Support Estimates, World Bank WDI/CWON, WTO ePing/IDB, RAM Legacy Stock Assessment Database, and Natural Earth), datasets are excluded from version control and ignored via `.gitignore`. All aggregated summary benchmarks, regression coefficients, calibration diagnostics, ablation outputs, and Sobol indices needed to evaluate the findings are reported in the manuscript, Extended Data, and Supplementary Information.
>
> Researchers wishing to reconstruct the full empirical panels and re-execute the pipeline locally can acquire the underlying public datasets directly from the official institutional sources listed below.

---

## Primary Institutional Data Sources

To execute the empirical loader and spatial mapping modules locally, assemble the following open-access public data streams into `data/raw/` and `data/geo/`:

1. **Satellite Vessel Effort & Offshore Transshipment** — **Global Fishing Watch v3.0** apparent fishing hours, foreign distant-water fishing shares, carrier transshipment events, and untracked industrial vessel hours ([globalfishingwatch.org](https://globalfishingwatch.org)).
2. **Global Trade & Trophic Levels** — **UN Comtrade** bilateral seafood trade flows ([comtradeplus.un.org](https://comtradeplus.un.org)), **FAO FishStatJ** capture production ([fao.org/fishery/en/statistics/software/fishstatj](https://www.fao.org/fishery/en/statistics/software/fishstatj)), and **FishBase** species-group trophic levels ([fishbase.org](https://www.fishbase.org)).
3. **Fisheries Subsidies & Macro-Environmental Accounts** — **OECD Fisheries Support Estimates** ([stats.oecd.org](https://stats.oecd.org)) and global capacity-enhancing versus beneficial subsidy ledgers; **World Bank Open Data** and *The Changing Wealth of Nations* mangrove blue-carbon accounts ([databank.worldbank.org](https://databank.worldbank.org)); **WTO Most-Favoured-Nation** tariff and ePing non-tariff notification archives ([eping.wto.org](https://eping.wto.org)).
4. **Stock Status & Coastal Boundaries** — **RAM Legacy Stock Assessment Database v4.44 / v4.65** (`B/Bmsy` and `F/Fmsy` across 1,372 stocks; [ramlegacy.org](https://www.ramlegacy.org)) and **Natural Earth 1:50m** sovereign country boundaries ([naturalearthdata.com](https://www.naturalearthdata.com)).
5. **Ocean Climate Forcing Parameters** — Published CMIP6 decadal coastal shelf warming, deoxygenation, marine heatwave, and net primary productivity slopes for **GFDL-ESM4** (`MOM6-COBALTv2`) and **MPI-ESM1-2-HR** (`MPIOM TP04-HAMOCC6`) across `SSP1-2.6`, `SSP2-4.5`, and `SSP5-8.5` (parameterized internally in `model/cmip6_downscaler.py`).

Expected local directory layout when populated with downloaded datasets:

```text
ocean-extremes-fisher-debt-tipping-points/
├── data/
│   ├── raw/                          # Downloaded CSV inputs (excluded from git)
│   └── geo/
│       └── ne_50m_coastal_countries.geojson
```

---

## Core Module Inventory (15 Python Modules + ODD+D Specification)

### `model/` — Coastal Climate Forcing & Coupled Spatial Agent-Based Model (`ODD+D`)
1. **`model/cmip6_downscaler.py`** — Downscales published GFDL-ESM4 and MPI-ESM1-2-HR trends across `SSP1-2.6`, `SSP2-4.5`, and `SSP5-8.5`, superposing ENSO harmonics and $Q_{0.90}$ coastal marine heatwaves over a $0.1^\circ$ continental shelf lattice.
2. **`model/environment.py`** — Simulates the $10 \times 10$ ($0.1^\circ$) cross-shelf bathymetric grid, Arrhenius Metabolic Index ($\Phi$), hypoxic mortality, mangrove nursery subsidies, and CFL-bounded 5-point Laplacian fish diffusion.
3. **`model/agents.py`** — Implements five monthly interacting agent classes: small-scale fishers under Roy's safety-first debt-hazard switching, subsidized distant-water fleets (`logit`), intertidal land managers, trading hubs, and policy regulators.
4. **`model/engine.py`** — Integrates biophysical and behavioral states monthly with controlled causal ablation flags (`enable_debt_trap`, `enable_metabolic_compression`, `num_dwf`) and counterfactual policy rules.
5. **`model/calibration.py`** — Runs Pattern-Oriented Modeling (grid search + Nelder–Mead) against 2010–2020 global macro targets plus 2020–2024 out-of-sample country and 7-fold Leave-One-Region-Out cross-validation.
6. **`model/stochastic_evaluator.py`** — Executes $M = 30$ Common Random Numbers Monte Carlo policy ensembles (`2020–2045` under `SSP2-4.5`) across Status Quo, 30×30 MPA, WTO Subsidy Ban, Blue-Carbon Dividend, and Integrated reform.
- **`model/odd_protocol.md`** — Formal **ODD+D** (*Overview, Design concepts, Details + Decision-making*) model documentation.

### `analysis/` — Econometrics, Tipping Diagnostics, Ablation & Global Sensitivity
7. **`analysis/econometric_analysis.py`** — Estimates progressive OLS and 2012–2024 panel fixed-effects models with sovereign country-clustered standard errors ($N = 109$ states, $N = 1{,}417$ state-years) and covariate VIF checks.
8. **`analysis/compare_esm_models.py`** — Evaluates structural divergence between `GFDL-ESM4`, `MPI-ESM1-2-HR`, and multi-model mean projections across all emissions scenarios.
9. **`analysis/tipping_point_detector.py`** — Computes Gaussian-detrended variance, lag-1 autocorrelation ($\rho_1$), two-sided Kendall's $\tau$ early warning trends over a 48-month pre-collapse window, and harvest–biomass hysteresis integrals ($\mathcal{H}$).
10. **`analysis/mechanism_ablation.py`** — Executes paired 4-arm `SSP5-8.5` causal ablation experiments ($M = 30$ CRN seeds per arm) isolating household debt traps, metabolic compression, and subsidized DWF fleets.
11. **`analysis/surrogate_emulator.py`** — Generates $N_{\text{train}} = 512$ 30-year CRN Latin Hypercube ABM runs across 17 parameters and benchmarks standardized `MLPRegressor (128, 64)`, `HistGradientBoosting`, and `RandomForest` surrogates.
12. **`analysis/global_sensitivity_analysis.py`** — Evaluates $N = 73{,}728$ Saltelli Sobol first- and total-order sensitivity indices ($S_1, S_T, S_2$) validated directly against $304$ full 30-year mechanistic ABM simulations.

### `visualizations/` — Publication Figure Generators
13. **`visualizations/style_and_maps.py`** — Defines colorblind-safe palettes, Equal-Earth (`EPSG:8857`) choropleth mappers, co-registered $0.1^\circ$ shelf submaps, and raincloud helpers.
14. **`visualizations/generate_figures.py`** — Generates main Figures 1–6 (600 DPI PNG + vector PDF).
15. **`visualizations/generate_supplementary_figures.py`** — Generates Extended Data Figures 1–3 and Supplementary Figures S1–S2 (600 DPI PNG + vector PDF).

---

## Computational Environment & Execution

Verified under **Python 3.12**:

```bash
git clone https://github.com/SezginTunca/ocean-extremes-fisher-debt-tipping-points.git
cd ocean-extremes-fisher-debt-tipping-points

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Once local public inputs are placed in data/raw/ and data/geo/:
bash run_all.sh
```

---

## Citation & License

This replication codebase is released under the [MIT License](LICENSE).

```bibtex
@software{tunca_2026_ocean_extremes_fisher_debt,
  author       = {Tunca, Sezgin},
  title        = {{Replication Codebase: Ocean extremes and fisher debt drive tipping points in coastal blue economies}},
  year         = {2026},
  version      = {v1.0.0},
  url          = {https://github.com/SezginTunca/ocean-extremes-fisher-debt-tipping-points}
}
```
