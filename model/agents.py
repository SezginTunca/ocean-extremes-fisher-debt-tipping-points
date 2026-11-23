#!/usr/bin/env python3
"""
agents.py
Heterogeneous Socio-Economic Agents for ABM-Earth-TropEcS:
1. SmallScaleFisher (SSF): Roy's Safety-First debt-hazard trap & risk aversion
2. IndustrialDWFFleet (DWF): Capacity subsidies, RUM spatial choice, distant-water effort
3. AquacultureBlueCarbonManager: Mangrove conservation vs. intensive aquaculture trade-offs
4. TradingProcessingHub: Export margins, domestic food security, tariff barriers
5. PolicyRegulator: MPA enforcement, WTO subsidy prohibition, adaptive management
"""

import numpy as np
from typing import Dict, List, Tuple


class SmallScaleFisher:
    def __init__(
        self,
        agent_id: int,
        home_port: Tuple[int, int] = (0, 5),
        debt_threshold: float = 250.0,
        risk_aversion: float = 0.35,
        base_fuel_rate: float = 0.25,
        exit_patience_months: int = 24,
        enable_debt_trap: bool = True
    ):
        self.id = agent_id
        self.home_port = home_port
        self.debt_threshold = debt_threshold
        self.risk_aversion = risk_aversion
        self.base_fuel_rate = base_fuel_rate
        self.exit_patience = exit_patience_months
        self.enable_debt_trap = bool(enable_debt_trap)

        # Dynamic state
        self.debt = 0.0
        self.cumulative_income = 0.0
        self.is_active = True
        self.in_debt_trap = False
        self.insolvent_months = 0
        self.total_trips = 0
        self.risky_trips_taken = 0
        self.recent_catch = 0.0
        self.last_trip_distance = 0.0
        self.last_wave_hazard = 0.0
        self.last_net_income = 0.0
        self.trip_distances: List[float] = []
        self.wave_hazards: List[float] = []
        self.solvent_trip_distances: List[float] = []
        self.solvent_wave_hazards: List[float] = []
        self.trapped_trip_distances: List[float] = []
        self.trapped_wave_hazards: List[float] = []

    def decide_spatial_effort(
        self,
        env,
        fuel_price: float,
        ex_vessel_price: float,
        is_extreme_weather: bool,
        rng: np.random.Generator
    ) -> Tuple[int, int, float]:
        """
        Chooses fishing location (row, col) and trip effort.
        Structural micro-behavioral rule: under normal solvency, the fisher avoids extreme
        weather and weights distance/hypoxia risk proportionally to risk_aversion. When
        enable_debt_trap is active and cumulative informal debt exceeds debt_threshold
        (Roy's safety-first insolvency switch; Roy 1952; Chavas 2004), subsistence survival
        overrides storm avoidance and discounts physical risk by 95%. Macro-level fleet
        reallocation, offshore hazard exposure, and fishery hysteresis emerge endogenously.
        """
        if not self.is_active:
            self.last_trip_distance = 0.0
            self.last_wave_hazard = 0.0
            return -1, -1, 0.0

        # Check debt-hazard trap activation
        if self.enable_debt_trap and self.debt > self.debt_threshold:
            self.in_debt_trap = True
        else:
            self.in_debt_trap = False

        # Extreme weather aversion unless trapped
        if is_extreme_weather and not self.in_debt_trap:
            # Fisher stays in port to protect life and vessel, incurring household subsistence deficit
            self.debt += 4.5
            self.last_trip_distance = 0.0
            self.last_wave_hazard = 0.05
            self.last_net_income = -4.5
            if self.enable_debt_trap and self.debt > self.debt_threshold:
                self.insolvent_months += 1
                self.in_debt_trap = True
            return -1, -1, 0.0

        # Candidate fishing cells: nearshore columns 0 to 4
        # In debt trap, fisher expands search into deeper, riskier waters (cols 0 to 6)
        if self.in_debt_trap:
            self.risky_trips_taken += 1
            cand_cols = np.arange(0, min(6, env.w))
        else:
            cand_cols = np.arange(0, min(4, env.w))

        cand_rows = np.arange(env.h)

        best_u = -1e9
        best_cell = (0, 0)

        for r in cand_rows:
            for c in cand_cols:
                if env.is_mpa[r, c]:
                    continue  # Respect MPA boundary

                dist = env.distance_to_port[r, c]
                expected_catch = min(env.biomass[r, c] * (0.019 if self.in_debt_trap else 0.015), 0.22)
                expected_rev = expected_catch * ex_vessel_price * 60.0
                fuel_cost = dist * self.base_fuel_rate * fuel_price
                effort_cost = 3.5  # Fixed operational cost (bait, gear maintenance)

                # Risk penalty: variance increases with distance and hypoxia
                risk_penalty = self.risk_aversion * (dist * 0.4 + max(0.0, env.phi_crit - env.phi[r, c]) * 15.0)

                # Under Roy's safety-first rule, debt-trapped fishers heavily discount risk to chase revenue
                if self.in_debt_trap:
                    u = expected_rev - fuel_cost - effort_cost - 0.05 * risk_penalty + rng.normal(0, 0.6)
                else:
                    u = expected_rev - fuel_cost - effort_cost - risk_penalty + rng.normal(0, 0.6)

                if u > best_u:
                    best_u = u
                    best_cell = (r, c)

        self.total_trips += 1
        chosen_r, chosen_c = best_cell[0], best_cell[1]
        dist_chosen = float(env.distance_to_port[chosen_r, chosen_c])
        hyp_stress = max(0.0, float(env.phi_crit - env.phi[chosen_r, chosen_c]))
        wave_haz = float(np.clip(
            0.08 + 0.019 * dist_chosen + (0.24 if is_extreme_weather else 0.0) + 0.09 * hyp_stress,
            0.04, 0.96
        ))
        self.last_trip_distance = dist_chosen
        self.last_wave_hazard = wave_haz
        self.trip_distances.append(dist_chosen)
        self.wave_hazards.append(wave_haz)
        if self.in_debt_trap:
            self.trapped_trip_distances.append(dist_chosen)
            self.trapped_wave_hazards.append(wave_haz)
        else:
            self.solvent_trip_distances.append(dist_chosen)
            self.solvent_wave_hazards.append(wave_haz)
        return chosen_r, chosen_c, 1.0

    def update_financials(
        self,
        realized_catch: float,
        ex_vessel_price: float,
        fuel_cost: float,
        subsistence_cost: float = 3.2
    ):
        """
        Updates artisanal household cash-flow, subsistence balance, and informal micro-credit debt.
        Includes monthly household subsistence expenditure (food, shelter, gear amortization) and
        informal interest (0.4%/month) when indebted.
        """
        if not self.is_active:
            return

        self.recent_catch = realized_catch
        revenue = realized_catch * ex_vessel_price * 60.0
        trip_cost = fuel_cost + 3.5 + subsistence_cost
        net_income = revenue - trip_cost
        self.last_net_income = float(net_income)

        # Endogenous informal moneylender interest surcharge: r_loan(Debt_t) = r_0 + r_1 * tanh(Debt_t / Debt_thresh)
        if self.debt > 0:
            r_0 = 0.003
            r_1 = 0.002
            r_loan = r_0 + r_1 * float(np.tanh(self.debt / max(1.0, self.debt_threshold)))
            self.debt *= (1.0 + r_loan)

        if net_income >= 0:
            # Pay down debt first
            debt_payment = min(self.debt, net_income * 0.70)
            self.debt -= debt_payment
            self.cumulative_income += (net_income - debt_payment)
            self.insolvent_months = max(0, self.insolvent_months - 1)
        else:
            self.debt += abs(net_income)
            if self.debt > self.debt_threshold:
                self.insolvent_months += 1
                self.in_debt_trap = True

        if self.debt <= self.debt_threshold:
            self.in_debt_trap = False

        # Exit if insolvent for too long
        if self.insolvent_months >= self.exit_patience:
            self.is_active = False

    def receive_dividend(self, dividend_amount: float):
        """Receives blue carbon dividend or social safety net transfer."""
        if not self.is_active or dividend_amount <= 0:
            return
        relief = min(self.debt, dividend_amount)
        self.debt -= relief
        remaining = dividend_amount - relief
        self.cumulative_income += remaining
        if self.debt <= self.debt_threshold:
            self.in_debt_trap = False
            self.insolvent_months = max(0, self.insolvent_months - 2)


class IndustrialDWFFleet:
    def __init__(
        self,
        fleet_id: int,
        capacity_subsidy_rate: float = 0.35,  # sigma: fuel subsidy fraction
        vessel_count: int = 15,
        catchability: float = 0.035,
        base_fuel_rate: float = 4.5
    ):
        self.id = fleet_id
        self.subsidy_rate = capacity_subsidy_rate
        self.vessels = vessel_count
        self.q = catchability
        self.base_fuel_rate = base_fuel_rate
        self.total_landings = 0.0
        self.total_profit = 0.0

    def allocate_spatial_effort(
        self,
        env,
        fuel_price: float,
        ex_vessel_price: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Random Utility Model (RUM) allocating vessel effort across offshore and shelf grid cells.
        Capacity-enhancing fuel subsidies (sigma) lower effective fuel costs and expand active
        distant-water vessel deployments along the shelf break and high seas.
        """
        effort_matrix = np.zeros((env.h, env.w))
        effective_fuel_price = fuel_price * (1.0 - self.subsidy_rate)

        utilities = np.full((env.h, env.w), -1e9)
        # DWF operates primarily in shelf and offshore waters (cols 2 to 9)
        for r in range(env.h):
            for c in range(2, env.w):
                if env.is_mpa[r, c]:
                    continue
                cpue = env.biomass[r, c] * self.q
                dist = env.distance_to_port[r, c]
                cost = dist * self.base_fuel_rate * effective_fuel_price
                expected_rev = cpue * ex_vessel_price * 10.0
                utilities[r, c] = expected_rev - cost

        # Softmax spatial choice
        flat_u = utilities.flatten()
        valid = flat_u > -1e8
        if not np.any(valid):
            return effort_matrix

        exp_u = np.exp((flat_u - np.max(flat_u[valid])) / 35.0) * valid
        probs = exp_u / np.sum(exp_u)

        # Subsidy- and fuel-price-elastic active vessel-trip deployments
        subsidy_effort_multiplier = (0.70 + 0.90 * self.subsidy_rate) / (1.0 + 0.25 * max(0.0, effective_fuel_price - 0.65))
        active_deployments = max(1, int(round(self.vessels * subsidy_effort_multiplier)))

        chosen_cells = rng.choice(len(probs), size=active_deployments, p=probs)
        for cell_idx in chosen_cells:
            r, c = divmod(cell_idx, env.w)
            effort_matrix[r, c] += 1.0

        return effort_matrix


class AquacultureBlueCarbonManager:
    def __init__(
        self,
        total_coastal_area_ha: float = 5000.0,
        mangrove_fraction: float = 0.50,
        carbon_price_usd_per_tco2: float = 25.0,
        aqua_yield_t_per_ha: float = 4.0,
        aqua_profit_usd_per_t: float = 350.0
    ):
        self.total_area = total_coastal_area_ha
        self.initial_mangrove_frac = float(mangrove_fraction)
        self.mangrove_frac = float(mangrove_fraction)
        self.carbon_price = carbon_price_usd_per_tco2
        self.aqua_yield = aqua_yield_t_per_ha
        self.aqua_profit = aqua_profit_usd_per_t
        self.cumulative_carbon_revenue = 0.0
        self.cumulative_aqua_profit = 0.0

    def step_land_allocation(self, carbon_price_t: float, mpa_enforced: bool, mpa_fraction: float = 0.10):
        """
        Trades off intensive aquaculture pond conversion vs. mangrove blue carbon preservation.
        Higher carbon price, coastal MPA protection, and baseline mangrove endowment shift land
        toward mangrove restoration.
        """
        self.carbon_price = carbon_price_t
        # Annual return per hectare:
        # Mangrove: ~12 tCO2e/ha/year sequestration * carbon_price + coastal protection & MPA co-management premium
        mpa_premium = (520.0 * (mpa_fraction / 0.30)) if mpa_enforced else (180.0 * (mpa_fraction / 0.10))
        carbon_return_per_ha = 12.0 * self.carbon_price + 140.0 + mpa_premium
        aqua_return_per_ha = self.aqua_yield * self.aqua_profit

        econ_target = carbon_return_per_ha / (carbon_return_per_ha + aqua_return_per_ha + 1e-9)
        target_frac = np.clip(econ_target + 0.35 * (self.initial_mangrove_frac - 0.50), 0.12, 0.92)

        # Smooth transition (coastal restoration / pond conversion rate)
        self.mangrove_frac += 0.05 * (target_frac - self.mangrove_frac)

        mangrove_ha = self.total_area * self.mangrove_frac
        aqua_ha = self.total_area * (1.0 - self.mangrove_frac)

        monthly_c_rev = (mangrove_ha * 12.0 * self.carbon_price) / 12.0
        monthly_aq_prof = (aqua_ha * aqua_return_per_ha) / 12.0

        self.cumulative_carbon_revenue += monthly_c_rev
        self.cumulative_aqua_profit += monthly_aq_prof

        return {
            "mangrove_hectares": mangrove_ha,
            "aquaculture_hectares": aqua_ha,
            "monthly_carbon_revenue": monthly_c_rev,
            "monthly_aquaculture_profit": monthly_aq_prof
        }


class TradingProcessingHub:
    def __init__(
        self,
        export_tariff_pct: float = 8.5,
        ntm_cost_pct: float = 5.0,
        baseline_ex_vessel_price: float = 3.2,
        export_price_multiplier: float = 2.4
    ):
        self.tariff_pct = export_tariff_pct
        self.ntm_pct = ntm_cost_pct
        self.ex_vessel_price = baseline_ex_vessel_price
        self.export_mult = export_price_multiplier
        self.total_domestic_biomass = 0.0
        self.total_exported_biomass = 0.0

    def process_landings(self, total_landings: float) -> Dict[str, float]:
        """
        Allocates landed biomass between domestic food market (nutrition security)
        and international seafood export markets.
        Higher trade frictions (tariffs + NTMs) divert biomass into domestic supply.
        """
        friction = (self.tariff_pct + self.ntm_pct) / 100.0
        # Friction diminishes export attractiveness
        export_share = np.clip(0.65 - 0.8 * friction, 0.15, 0.85)

        export_tonnes = total_landings * export_share
        domestic_tonnes = total_landings * (1.0 - export_share)

        export_price = self.ex_vessel_price * self.export_mult * (1.0 - friction)
        export_revenue = export_tonnes * export_price * 1000.0

        self.total_domestic_biomass += domestic_tonnes
        self.total_exported_biomass += export_tonnes

        return {
            "domestic_biomass_tonnes": domestic_tonnes,
            "exported_biomass_tonnes": export_tonnes,
            "export_revenue_usd": export_revenue,
            "export_share": export_share
        }


class PolicyRegulator:
    def __init__(
        self,
        target_mpa_fraction: float = 0.30,
        wto_subsidy_prohibition: bool = False,
        social_safety_net_active: bool = False
    ):
        self.target_mpa = target_mpa_fraction
        self.wto_prohibition = wto_subsidy_prohibition
        self.safety_net = social_safety_net_active
        self.total_transfers = 0.0

    def evaluate_emergency_transfer(self, fishers: List[SmallScaleFisher], fuel_price: float):
        """
        Deploys targeted social safety net transfers to fishers trapped in debt
        during extreme compound events, preventing catastrophic bankruptcy.
        """
        if not self.safety_net:
            return 0.0

        transfers = 0.0
        for f in fishers:
            if f.is_active and (f.in_debt_trap or f.debt > 0.6 * f.debt_threshold):
                relief = min(f.debt, 165.0)
                f.debt -= relief
                transfers += relief
                if f.debt <= f.debt_threshold:
                    f.in_debt_trap = False

        self.total_transfers += transfers
        return transfers

    def disburse_blue_carbon_dividend(
        self,
        fishers: List[SmallScaleFisher],
        carbon_revenue: float,
        dividend_share: float = 0.50
    ) -> float:
        """
        Disburses community blue carbon dividends directly to local artisanal fishers
        funded from mangrove carbon credits, supporting livelihood security and debt alleviation.
        """
        if not self.safety_net or carbon_revenue <= 0:
            return 0.0

        active_fishers = [f for f in fishers if f.is_active]
        if not active_fishers:
            return 0.0

        total_payout = carbon_revenue * dividend_share
        per_fisher = total_payout / len(active_fishers)

        for f in active_fishers:
            f.receive_dividend(per_fisher)

        self.total_transfers += total_payout
        return total_payout
