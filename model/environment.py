#!/usr/bin/env python3
"""
environment.py
Downscaled Coastal Interface & Biophysical Simulation for ABM-Earth-TropEcS.
Implements:
- 2D coastal grid interface (Inshore, Shelf, High Seas, MPA zones, Mangrove fringes)
- Coupled Deutsch et al. (2020, Nature) Metabolic Index (Phi)
- Dynamic biomass redistribution driven by thermal-oxygen habitat compression
- Biomass growth with carrying capacity K and growth rate r(Phi)
- Mangrove blue carbon sequestration and coastal protection attenuation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class CoastalEnvironment:
    def __init__(
        self,
        grid_width: int = 10,
        grid_height: int = 10,
        r_growth: float = 0.45,
        carrying_capacity_k: float = 1000.0,
        phi_crit: float = 2.0,
        phi_opt: float = 4.0,
        diffusivity: float = 0.15,
        mpa_fraction: float = 0.10,
        mangrove_fraction: float = 0.15,
        carbon_seq_rate: float = 3.5,  # tCO2e / ha / year
        seed: int = 42
    ):
        self.w = grid_width
        self.h = grid_height
        self.r_growth = r_growth
        self.k_capacity = carrying_capacity_k
        self.phi_crit = phi_crit
        self.phi_opt = phi_opt
        self.diffusivity = diffusivity
        self.mpa_fraction = mpa_fraction
        self.mangrove_fraction = mangrove_fraction
        self.carbon_seq_rate = carbon_seq_rate
        self.rng = np.random.default_rng(seed)

        # Grid layout: Column 0 = Inshore/Mangrove, Column 1-4 = Nearshore/Shelf, Column 5-9 = Offshore/High Seas
        # Row 0-9 distance to central port located at (0, 5)
        self.port_x, self.port_y = 0, self.h // 2

        # Distance to port matrix (nautical miles approx)
        x_coords, y_coords = np.meshgrid(np.arange(self.w), np.arange(self.h))
        self.distance_to_port = np.sqrt((x_coords - self.port_x)**2 + (y_coords - self.port_y)**2) * 5.0  # 5 nm per cell

        # Depth gradient (meters): 10m inshore to 410m offshore
        self.depth = 10.0 + (x_coords / max(1, (self.w - 1))) * 400.0

        # 2D Cross-shelf & along-shore bathymetric/estuarine thermal & oxygen offsets
        # Nearshore estuarine/upwelling cells experience higher thermal anomalies and benthic OMZ shoaling
        shelf_t = np.linspace(1.1, -0.5, self.w)[np.newaxis, :]
        lat_t = np.linspace(0.35, -0.35, self.h)[:, np.newaxis]
        self.spatial_sst_offset = shelf_t + lat_t

        shelf_do = np.linspace(-2.1, 1.6, self.w)[np.newaxis, :]
        lat_do = np.linspace(-0.65, 0.65, self.h)[:, np.newaxis]
        self.spatial_do_offset = shelf_do + lat_do

        # Designated Zones: Deterministic/Seeded MPA placement in nearshore-shelf productive zone
        self.is_mpa = np.zeros((self.h, self.w), dtype=bool)
        n_mpa_cells = int(round(self.w * self.h * self.mpa_fraction))
        if n_mpa_cells > 0:
            candidate_cells = [(r, c) for c in range(1, self.w - 1) for r in range(self.h)]
            n_pick = min(n_mpa_cells, len(candidate_cells))
            mpa_indices = self.rng.choice(len(candidate_cells), size=n_pick, replace=False)
            for idx in mpa_indices:
                r, c = candidate_cells[idx]
                self.is_mpa[r, c] = True

        # Mangrove fringes along column 0 (inshore)
        self.mangrove_cover = np.zeros((self.h, self.w))
        n_mangrove_cells = int(round(self.h * np.clip(self.mangrove_fraction, 0.0, 1.0)))
        mangrove_rows = self.rng.choice(np.arange(self.h), size=min(max(1, n_mangrove_cells), self.h), replace=False)
        self.mangrove_cover[mangrove_rows, 0] = 1.0

        # Initial biomass distribution: balanced around 70% of K
        self.biomass = np.ones((self.h, self.w)) * (self.k_capacity / (self.w * self.h)) * 0.70

        # Current biophysical state
        self.sst = np.ones((self.h, self.w)) * 24.0
        self.po2 = np.ones((self.h, self.w)) * 17.5
        self.phi = np.ones((self.h, self.w)) * self.phi_opt
        self.npp = np.ones((self.h, self.w)) * 1.0
        self.cumulative_blue_carbon = 0.0

    def update_climate_forcing(self, sst_val: float, po2_val: float, npp_val: float, phi_val: float):
        """Applies downscaled regional ESM climate forcing across the 2D coastal grid."""
        self.sst = sst_val + self.spatial_sst_offset
        self.po2 = np.maximum(1.0, po2_val + self.spatial_do_offset)

        # Recalculate local Metabolic Index Phi using Deutsch et al. (2020, Nature) Arrhenius formulation:
        # Phi(T, pO2) = A0 * pO2 / exp(-E_0 / (k_B * T_Kelvin))
        kB = 8.617333262145e-5   # Boltzmann constant (eV/K)
        E_act = 0.45              # Activation energy (eV)
        T_ref_k = 18.0 + 273.15
        pO2_ref = 18.0
        A0 = self.phi_opt * np.exp(-E_act / (kB * T_ref_k)) / pO2_ref

        T_kelvin = self.sst + 273.15
        self.phi = A0 * self.po2 / np.exp(-E_act / (kB * T_kelvin))
        self.phi = np.maximum(0.1, self.phi)

        # NPP distributed higher nearshore due to coastal runoff, mangrove detritus, and shelf upwelling
        mangrove_nursery_boost = 1.0 + 0.25 * np.mean(self.mangrove_cover[:, 0])
        npp_gradient = np.linspace(1.40 * mangrove_nursery_boost, 0.80, self.w)
        self.npp = np.broadcast_to(npp_val * npp_gradient[np.newaxis, :], (self.h, self.w)).copy()

    def step_biological_dynamics(self, harvest_matrix: np.ndarray) -> Dict[str, float]:
        """
        Updates fish stock dynamics:
        1. Harvest extraction
        2. Growth modulated by Metabolic Index: r_eff = r * max(0, (Phi - Phi_crit)/(Phi_opt - Phi_crit))
           plus aerobic scope collapse mortality when Phi < Phi_crit
        3. Carrying capacity K modulated by NPP
        4. Spatial diffusion with zero-flux Neumann boundaries (strictly mass-conserving when r_growth=0, harvest=0)
        """
        # Step 1: Harvest removal (cannot exceed current biomass, leaving 0.1 minimum refuge)
        available_biomass = np.maximum(0.0, self.biomass - 0.1)
        realized_harvest = np.minimum(available_biomass, np.maximum(0.0, harvest_matrix))
        self.biomass -= realized_harvest

        # Step 2: Growth & Hypoxic Compression Stress
        if self.r_growth > 0.0:
            phi_factor = np.clip((self.phi - self.phi_crit) / (self.phi_opt - self.phi_crit + 1e-9), 0.12, 1.5)
            r_monthly = (self.r_growth / 12.0) * phi_factor
            k_local = (self.k_capacity / (self.w * self.h)) * self.npp

            growth = r_monthly * self.biomass * (1.0 - self.biomass / np.maximum(1.0, k_local))
            # Hypoxic metabolic stress penalty where Phi < Phi_crit
            hypoxic_deficit = np.maximum(0.0, self.phi_crit - self.phi) / max(0.5, self.phi_crit)
            hypoxic_loss = 0.008 * hypoxic_deficit * self.biomass
            self.biomass = np.maximum(0.1, self.biomass + growth - hypoxic_loss)

        # Step 3: Spatial Diffusion with Zero-Flux Neumann Boundary Conditions
        # Enforce CFL stability condition for 2D explicit scheme: D <= 0.20
        cfl_diffusivity = min(0.20, max(0.01, self.diffusivity))
        padded = np.pad(self.biomass, pad_width=1, mode="edge")
        laplacian = (
            padded[2:, 1:-1] + padded[:-2, 1:-1] +
            padded[1:-1, 2:] + padded[1:-1, :-2] -
            4.0 * self.biomass
        )
        max_cell_capacity = 2.5 * (self.k_capacity / (self.w * self.h))
        self.biomass = np.clip(self.biomass + cfl_diffusivity * laplacian, 0.1, max_cell_capacity)

        # Step 4: Blue Carbon accumulation from mangrove coverage
        # Convert cell coverage to hectare equivalent (each inshore cell represents total_area_ha / h)
        monthly_carbon = np.sum(self.mangrove_cover) * (self.carbon_seq_rate / 12.0)
        self.cumulative_blue_carbon += monthly_carbon

        total_biomass = float(np.sum(self.biomass))
        mean_phi = float(np.mean(self.phi))
        hypoxic_cells = int(np.sum(self.phi < self.phi_crit))

        return {
            "total_biomass": total_biomass,
            "total_harvest": float(np.sum(realized_harvest)),
            "mean_phi": mean_phi,
            "hypoxic_fraction": hypoxic_cells / float(self.w * self.h),
            "monthly_blue_carbon": float(monthly_carbon),
            "cumulative_blue_carbon": float(self.cumulative_blue_carbon)
        }

    def update_mangrove_coverage(self, mangrove_ha: float, total_area_ha: float = 5000.0):
        """Dynamically updates inshore mangrove grid coverage from land manager decisions."""
        frac = np.clip(mangrove_ha / (total_area_ha + 1e-9), 0.05, 1.0)
        self.mangrove_cover[:, 0] = frac
