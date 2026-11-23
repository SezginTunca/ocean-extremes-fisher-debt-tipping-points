#!/usr/bin/env python3
"""
engine.py
Coupled Simulation Engine for ABM-Earth-TropEcS.
Executes monthly time steps (2000-2050), coupling ESM downscaled climate drivers,
biophysical stock dynamics, heterogeneous agent decisions, and market telecoupling.
Logs system state trajectory for tipping point analysis and surrogate training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

try:
    from .environment import CoastalEnvironment
    from .agents import (
        SmallScaleFisher,
        IndustrialDWFFleet,
        AquacultureBlueCarbonManager,
        TradingProcessingHub,
        PolicyRegulator
    )
except ImportError:
    from environment import CoastalEnvironment
    from agents import (
        SmallScaleFisher,
        IndustrialDWFFleet,
        AquacultureBlueCarbonManager,
        TradingProcessingHub,
        PolicyRegulator
    )

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from cmip6_downscaler import CMIP6CoastalDownscaler


class ABMEarthEngine:
    def __init__(
        self,
        scenario: str = "SSP2-4.5",
        start_year: int = 2000,
        end_year: int = 2050,
        num_ssf_fishers: int = 40,
        num_dwf_vessels: int = 15,
        capacity_subsidy_rate: float = 0.35,
        base_fuel_price: float = 1.0,
        fuel_price_volatility: float = 0.20,
        carbon_price: float = 25.0,
        export_tariff_pct: float = 8.5,
        mpa_fraction: float = 0.10,
        phi_crit: float = 2.0,
        r_growth: float = 0.45,
        esm_model: str = "GFDL-ESM4",
        wto_prohibition: bool = False,
        safety_net_active: bool = False,
        risk_aversion: float = 0.35,
        debt_threshold: float = 250.0,
        diffusivity: float = 0.15,
        ex_vessel_price_base: float = 3.2,
        sst_warming_rate: Optional[float] = None,
        carrying_capacity_k: float = 1000.0,
        mangrove_fraction: float = 0.50,
        seed: int = 42,
        enable_debt_trap: bool = True,
        enable_metabolic_compression: bool = True
    ):
        self.scenario = scenario
        self.esm_model = esm_model
        self.start_year = start_year
        self.end_year = end_year
        self.enable_debt_trap = bool(enable_debt_trap)
        self.enable_metabolic_compression = bool(enable_metabolic_compression)
        self.rng = np.random.default_rng(seed)

        # Policy & Economic switches
        self.initial_capacity_subsidy_rate = float(capacity_subsidy_rate)
        self.capacity_subsidy_rate = float(capacity_subsidy_rate)
        self.base_fuel_price = float(base_fuel_price)
        self.fuel_volatility = float(fuel_price_volatility)
        self.carbon_price = float(carbon_price)
        self.export_tariff_pct = float(export_tariff_pct)
        self.mpa_fraction = float(mpa_fraction)
        self.mangrove_fraction = float(mangrove_fraction)
        self.wto_prohibition = bool(wto_prohibition)
        self.safety_net_active = bool(safety_net_active)
        self.ex_vessel_price_base = float(ex_vessel_price_base)
        self.r_growth = float(r_growth)
        self.phi_crit = float(phi_crit)
        self.diffusivity = float(diffusivity)
        self.risk_aversion = float(risk_aversion)
        self.debt_threshold = float(debt_threshold)
        self.num_ssf = int(round(num_ssf_fishers))
        self.num_dwf = int(round(num_dwf_vessels))

        # Initialize Climate Downscaler
        self.downscaler = CMIP6CoastalDownscaler(esm_model=esm_model, seed=seed)
        self.climate_data = self.downscaler.generate_coastal_forcing(
            scenario=scenario,
            start_year=start_year,
            end_year=end_year,
            phi_crit=phi_crit,
            sst_warming_rate=sst_warming_rate
        )

        # Initialize Environment
        self.env = CoastalEnvironment(
            r_growth=self.r_growth,
            carrying_capacity_k=carrying_capacity_k,
            phi_crit=self.phi_crit,
            diffusivity=self.diffusivity,
            mpa_fraction=self.mpa_fraction,
            mangrove_fraction=self.mangrove_fraction,
            seed=seed
        )

        # Initialize Agents with realistic heterogeneous initial debt and risk preferences
        self.ssf_fishers = []
        for i in range(self.num_ssf):
            q_frac = i / max(1, self.num_ssf - 1)
            f = SmallScaleFisher(
                agent_id=i,
                debt_threshold=self.debt_threshold * (0.75 + 0.50 * q_frac),
                risk_aversion=max(0.05, self.risk_aversion + self.rng.uniform(-0.08, 0.08)),
                exit_patience_months=600,
                enable_debt_trap=self.enable_debt_trap
            )
            # Initial informal gear/vessel working-capital debt distribution across coastal households
            f.debt = float((0.15 + 0.72 * (1.0 - q_frac)) * f.debt_threshold)
            self.ssf_fishers.append(f)

        self.dwf_fleet = IndustrialDWFFleet(
            fleet_id=1,
            capacity_subsidy_rate=self.capacity_subsidy_rate,
            vessel_count=self.num_dwf
        )

        self.blue_carbon_mgr = AquacultureBlueCarbonManager(
            carbon_price_usd_per_tco2=self.carbon_price,
            mangrove_fraction=self.mangrove_fraction
        )

        self.trading_hub = TradingProcessingHub(
            export_tariff_pct=self.export_tariff_pct,
            baseline_ex_vessel_price=self.ex_vessel_price_base
        )

        self.regulator = PolicyRegulator(
            target_mpa_fraction=self.mpa_fraction,
            wto_subsidy_prohibition=self.wto_prohibition,
            social_safety_net_active=self.safety_net_active
        )

        self.spatial_effort_ssf = np.zeros((self.env.h, self.env.w), dtype=float)
        self.spatial_effort_dwf = np.zeros((self.env.h, self.env.w), dtype=float)
        self.spatial_effort_total = np.zeros((self.env.h, self.env.w), dtype=float)
        self.spatial_biomass_end = self.env.biomass.copy()
        self.history = []

    def run_simulation(self) -> pd.DataFrame:
        """Executes full monthly time loop from start_year to end_year."""
        n_steps = len(self.climate_data)

        for step in range(n_steps):
            row_clim = self.climate_data.iloc[step]
            year = int(row_clim["year"])
            month = int(row_clim["month"])

            # Policy transition triggers (WTO subsidy prohibition activated starting in 2025)
            if self.wto_prohibition and year >= 2025:
                self.dwf_fleet.subsidy_rate = 0.0
            else:
                self.dwf_fleet.subsidy_rate = self.initial_capacity_subsidy_rate

            # Dynamic fuel price with volatility and stochastic shocks
            fuel_shock = self.rng.lognormal(0, self.fuel_volatility)
            current_fuel_price = self.base_fuel_price * fuel_shock

            # Ex-vessel price influenced by global market demand and local scarcity
            current_biomass_level = float(np.sum(self.env.biomass))
            ex_vessel_price = self.ex_vessel_price_base * (750.0 / max(150.0, current_biomass_level))**0.10

            # 1. Update climate forcing (fixed baseline when metabolic compression ablation is disabled)
            if self.enable_metabolic_compression:
                self.env.update_climate_forcing(
                    sst_val=row_clim["sst_celsius"],
                    po2_val=row_clim["po2_kpa"],
                    npp_val=row_clim["npp_index"],
                    phi_val=row_clim["metabolic_index_phi"]
                )
            else:
                base_clim = self.climate_data.iloc[0]
                self.env.update_climate_forcing(
                    sst_val=base_clim["sst_celsius"],
                    po2_val=base_clim["po2_kpa"],
                    npp_val=1.0,
                    phi_val=base_clim["metabolic_index_phi"]
                )

            is_extreme_event = bool(row_clim["compound_shock"] == 1)

            # 2. Small-Scale Fishers effort allocation
            ssf_harvest_matrix = np.zeros((self.env.h, self.env.w))

            # Industrial DWF competition & hypoxic compression reduce artisanal catch efficiency nearshore
            dwf_competition_drag = 1.0 / (1.0 + 0.40 * (self.num_dwf / 15.0) * (0.25 + 1.40 * self.dwf_fleet.subsidy_rate))
            # Community blue carbon dividends substitute for extractive fishing pressure via mangrove stewardship
            stewardship_effort_mod = 0.72 if self.safety_net_active else 1.0

            for idx_f, f in enumerate(self.ssf_fishers):
                if not f.is_active:
                    continue

                q_frac = idx_f / max(1, self.num_ssf - 1)
                r, c, effort = f.decide_spatial_effort(
                    env=self.env,
                    fuel_price=current_fuel_price,
                    ex_vessel_price=ex_vessel_price,
                    is_extreme_weather=is_extreme_event,
                    rng=self.rng
                )

                if r >= 0 and c >= 0:
                    self.spatial_effort_ssf[r, c] += effort
                    self.spatial_effort_total[r, c] += effort
                    local_phi_mod = np.clip(self.env.phi[r, c] / max(1.0, self.env.phi_crit + 0.35), 0.55, 1.15)
                    skill_factor = 0.65 + 0.95 * q_frac
                    catch_noise = self.rng.lognormal(-0.02, 0.15)
                    # Desperate fishers in debt trap intensify extraction rate (Roy's safety-first overfishing)
                    effort_intensity = 0.0125 if f.in_debt_trap else (0.0092 * stewardship_effort_mod)
                    catch = min(self.env.biomass[r, c] * effort_intensity * dwf_competition_drag * local_phi_mod * skill_factor * catch_noise, 0.18)
                    ssf_harvest_matrix[r, c] += catch
                    dist = self.env.distance_to_port[r, c]
                    fuel_cost = dist * f.base_fuel_rate * current_fuel_price
                    # Net household subsistence requirement after non-fishing livelihood buffer (which scales with q_frac)
                    net_subsistence = (1.6 - 5.2 * (q_frac ** 1.1)) + 1.1 * max(0.0, current_fuel_price - 0.9) + (0.8 if row_clim["is_mhw"] == 1 else 0.0)
                    f.update_financials(catch, ex_vessel_price, fuel_cost, subsistence_cost=net_subsistence)

            # 3. Industrial DWF Fleet effort allocation
            dwf_effort_matrix = self.dwf_fleet.allocate_spatial_effort(
                env=self.env,
                fuel_price=current_fuel_price,
                ex_vessel_price=ex_vessel_price,
                rng=self.rng
            )
            self.spatial_effort_dwf += dwf_effort_matrix
            self.spatial_effort_total += dwf_effort_matrix
            dwf_intensity = 0.13 + 0.18 * self.dwf_fleet.subsidy_rate
            dwf_harvest_matrix = np.minimum(self.env.biomass * 0.035, dwf_effort_matrix * dwf_intensity)

            # 4. Total harvest and environment step
            total_harvest_matrix = ssf_harvest_matrix + dwf_harvest_matrix
            bio_res = self.env.step_biological_dynamics(total_harvest_matrix)

            # 5. Blue Carbon & Aquaculture land allocation
            carbon_res = self.blue_carbon_mgr.step_land_allocation(
                carbon_price_t=self.carbon_price,
                mpa_enforced=(self.mpa_fraction >= 0.20),
                mpa_fraction=self.mpa_fraction
            )
            # Dynamically update environment mangrove coverage
            self.env.update_mangrove_coverage(carbon_res["mangrove_hectares"], self.blue_carbon_mgr.total_area)

            # 6. Trading Hub processing & market distribution
            hub_res = self.trading_hub.process_landings(bio_res["total_harvest"])

            # 7. Regulator emergency relief and blue carbon dividend disbursement
            relief_transfers = self.regulator.evaluate_emergency_transfer(
                fishers=self.ssf_fishers,
                fuel_price=current_fuel_price
            )
            dividend_payout = self.regulator.disburse_blue_carbon_dividend(
                fishers=self.ssf_fishers,
                carbon_revenue=carbon_res["monthly_carbon_revenue"],
                dividend_share=0.45 if self.safety_net_active else 0.0
            )

            # 8. Log comprehensive monthly state (computed post-financial update & policy relief)
            active_fishers = [f for f in self.ssf_fishers if f.is_active]
            active_ssf_count = len(active_fishers)
            debt_trapped_count = sum(1 for f in active_fishers if f.in_debt_trap)
            exited_ssf_count = self.num_ssf - active_ssf_count
            # Distress fraction includes active fishers in debt trap plus insolvencies/exits relative to initial cohort
            distress_count = debt_trapped_count + exited_ssf_count
            mean_debt = float(np.mean([f.debt for f in active_fishers])) if active_fishers else float(self.debt_threshold)
            total_ssf_harvest = float(np.sum(ssf_harvest_matrix))
            total_dwf_harvest = float(np.sum(dwf_harvest_matrix))

            # Micro-distributional income & trip risk metrics
            if active_fishers:
                incomes = np.array([f.last_net_income for f in active_fishers], dtype=float)
                p10, p25, p50, p75, p90 = np.percentile(incomes, [10, 25, 50, 75, 90])
                # Shifted positive wealth/income proxy for Gini index
                pos_inc = np.maximum(0.1, incomes - np.min(incomes) + 1.0)
                pos_inc_sorted = np.sort(pos_inc)
                n_inc = len(pos_inc_sorted)
                idx_arr = np.arange(1, n_inc + 1)
                ssf_gini = float((2.0 * np.sum(idx_arr * pos_inc_sorted) / (n_inc * np.sum(pos_inc_sorted))) - (n_inc + 1.0) / n_inc) if n_inc > 1 else 0.0
            else:
                p10 = p25 = p50 = p75 = p90 = 0.0
                ssf_gini = 0.0

            solvent_fishers = [f for f in active_fishers if not f.in_debt_trap and f.last_trip_distance > 0]
            trapped_fishers = [f for f in active_fishers if f.in_debt_trap and f.last_trip_distance > 0]
            mean_dist_solv = float(np.mean([f.last_trip_distance for f in solvent_fishers])) if solvent_fishers else 0.0
            mean_dist_trap = float(np.mean([f.last_trip_distance for f in trapped_fishers])) if trapped_fishers else 0.0
            mean_haz_solv = float(np.mean([f.last_wave_hazard for f in solvent_fishers])) if solvent_fishers else 0.0
            mean_haz_trap = float(np.mean([f.last_wave_hazard for f in trapped_fishers])) if trapped_fishers else 0.0

            self.history.append({
                "year": year,
                "month": month,
                "scenario": self.scenario,
                "esm_model": self.esm_model,
                "sst_celsius": row_clim["sst_celsius"],
                "metabolic_index_phi": bio_res["mean_phi"],
                "hypoxic_fraction": bio_res["hypoxic_fraction"],
                "is_mhw": row_clim["is_mhw"],
                "compound_shock": row_clim["compound_shock"],
                "fuel_price": current_fuel_price,
                "total_biomass": bio_res["total_biomass"],
                "total_harvest": bio_res["total_harvest"],
                "ssf_harvest": total_ssf_harvest,
                "dwf_harvest": total_dwf_harvest,
                "active_ssf_count": active_ssf_count,
                "exited_ssf_count": exited_ssf_count,
                "debt_trapped_ssf_count": debt_trapped_count,
                "debt_trap_fraction": distress_count / max(1, self.num_ssf),
                "mean_ssf_debt": mean_debt,
                "ssf_net_income_p10": float(p10),
                "ssf_net_income_p25": float(p25),
                "ssf_net_income_p50": float(p50),
                "ssf_net_income_p75": float(p75),
                "ssf_net_income_p90": float(p90),
                "ssf_income_gini": ssf_gini,
                "mean_trip_dist_solvent": mean_dist_solv,
                "mean_trip_dist_trapped": mean_dist_trap,
                "mean_hazard_solvent": mean_haz_solv,
                "mean_hazard_trapped": mean_haz_trap,
                "domestic_biomass_tonnes": hub_res["domestic_biomass_tonnes"],
                "exported_biomass_tonnes": hub_res["exported_biomass_tonnes"],
                "export_revenue_usd": hub_res["export_revenue_usd"],
                "mangrove_hectares": carbon_res["mangrove_hectares"],
                "aquaculture_hectares": carbon_res["aquaculture_hectares"],
                "monthly_carbon_revenue": carbon_res["monthly_carbon_revenue"],
                "monthly_aquaculture_profit": carbon_res["monthly_aquaculture_profit"],
                "cumulative_blue_carbon": bio_res["cumulative_blue_carbon"],
                "emergency_transfers_usd": relief_transfers,
                "blue_carbon_dividend_usd": dividend_payout
            })

        self.spatial_biomass_end = self.env.biomass.copy()
        df_out = pd.DataFrame(self.history)
        return df_out


if __name__ == "__main__":
    engine = ABMEarthEngine(scenario="SSP2-4.5")
    df_res = engine.run_simulation()
    print("Simulation Run Complete!")
    print(f"Total simulated months: {len(df_res)}")
    print(f"Final Biomass: {df_res['total_biomass'].iloc[-1]:.2f}")
    print(f"Peak Debt-Trap Fraction: {df_res['debt_trap_fraction'].max():.2f}")
    print(f"Total Blue Carbon Sequestered: {df_res['cumulative_blue_carbon'].iloc[-1]:.2f} tCO2e")
