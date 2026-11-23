# ODD+D Protocol Specification: ABM-Earth-TropEcS

**Model Title:** Agent-Based Model of Earth System Climate Forcing and Blue Economy Coastal Interfaces (ABM-Earth-TropEcS)  
**Standard Adherence:** ODD+D (Overview, Design concepts, Details + Human Decision) Protocol (Grimm et al., 2020; Müller et al., 2013) & SES behavioural ontology (Schlüter et al., 2017).  
**Authors:** Sezgin Tunca  
**Affiliation:** Department of Business and Sustainability, University of Southern Denmark  
**Conceptual Framework:** Structurally organized around the four multi-scale research pillars articulated in the TropEcS conceptual agenda (*Coupling Scales*, *Tipping Effects*, *Compound Events*, *Ecosystem Services & Biodiversity Policy*).  

---

## 1. Overview

### 1.1 Purpose
The purpose of ABM-Earth-TropEcS is to investigate the coupled social-ecological feedbacks between downscaled Earth System Model (ESM) biophysical stressors (marine heatwaves, deoxygenation, metabolic habitat compression) and human decision-making across global coastal and tropical interfaces. The model specifically evaluates:
1. **Pillar I (Coupling Scales):** How macro-scale climate anomalies downscaled to 0.1° coastal grids alter local species metabolic suitability and effort allocation.
2. **Pillar II (Exploring Tipping Effects):** Early warning signals of critical slowing down, biomass collapse, and negative social tipping cascades driven by fisher debt traps.
3. **Pillar III (Compound Events):** Emergence of extreme risk-taking behaviors when marine heatwaves, El Niño cycles, and global fuel price spikes co-occur.
4. **Pillar IV (Ecosystem Services & Policy):** Trade-offs between blue carbon mangrove preservation, intensive aquaculture, WTO subsidy prohibition, and 30×30 Marine Protected Area (MPA) networks.

### 1.2 Entities, State Variables, and Scales

#### Spatial Grid (Coastal Interface):
- **Domain:** 2D discrete cell grid ($N_x \times N_y = 10 \times 10$) representing cross-shelf coastal gradients from inshore mangrove fringes ($x=0$), nearshore shelf ($x=1..4$), to offshore high-seas waters ($x=5..9$).
- **Spatial Resolution:** Each cell represents approximately $0.1^\circ \times 0.1^\circ$ (~10 km resolution).
- **Temporal Resolution:** Monthly discrete time step ($\Delta t = 1 \text{ month}$), simulated from 2000 to 2050 (612 monthly steps).
- **Cell Variables:** Bathymetric depth ($z$), distance to home port ($D_{port}$), sea surface temperature ($SST$), bottom dissolved oxygen ($pO_2$), Metabolic Index ($\Phi$), Net Primary Productivity ($NPP$), fish biomass ($B_{x,y}$), MPA designation ($\mathcal{M}_{x,y} \in \{0, 1\}$), and mangrove canopy coverage ($M_{x,y} \in [0, 1]$).

#### Agent Entities:
1. **SmallScaleFisher (SSF):**
   - *State Variables:* Agent ID, home port coordinates, cumulative debt ($Debt_t$), debt-free status, debt-trap flag ($\mathcal{T}_{trap} \in \{0, 1\}$), risk aversion ($\lambda_{risk}$), fuel consumption rate ($f_{ssf}$), consecutive months in debt trap ($m_{ins}$).
2. **IndustrialDWFFleet (DWF):**
   - *State Variables:* Fleet ID, vessel count ($N_v$), catchability coefficient ($q_{dwf}$), capacity subsidy rate ($\sigma_{subsidy} \in [0, 1]$), base fuel consumption rate ($f_{dwf}$), cumulative landings, cumulative operational profit.
3. **AquacultureBlueCarbonManager:**
   - *State Variables:* Total managed intertidal area ($A_{total}$), mangrove conservation fraction ($\mu_{mang} \in [0, 1]$), carbon offset price ($P_{carbon}$), cumulative carbon sequestered ($C_{blue}$), cumulative aquaculture profit.
4. **TradingProcessingHub:**
   - *State Variables:* Applied MFN import/export tariff ($\tau_{mfn}$), non-tariff measure friction ($\theta_{ntm}$), ex-vessel price ($P_{ex}$), international export premium multiplier ($\gamma_{exp}$), domestic vs. export volume splits.
5. **PolicyRegulator:**
   - *State Variables:* MPA coverage target ($\Omega_{mpa}$), WTO capacity subsidy ban enforcement ($\mathcal{W}_{wto} \in \{0, 1\}$), emergency social safety net relief fund.

---

## 2. Design Concepts (ODD+D Human Decision-Making)

### 2.1 Theoretical Foundations
- **Metabolic Index ($\Phi$):** Grounded in Deutsch et al. (2020, *Nature*), quantifying the ratio of environmental oxygen supply to organismal metabolic demand:
  $$\Phi = A_{O_2} \frac{pO_2}{\exp(-E / (k_B T))}$$
  When $\Phi < \Phi_{crit}$, localized physiological hypoxia prevents sustained aerobic activity, driving bioclimatic avoidance or mortality.
- **Roy's Safety-First Criterion (Debt-Hazard Trap):** Small-scale fishers optimize expected utility under downside catastrophe constraints. If cumulative debt exceeds safety threshold $Debt_{thresh}$, the agent's risk tolerance inverts—accepting perilous sea states during El Niño/MHW compound shocks to stave off bankruptcy.
- **Random Utility Model (RUM):** Discrete spatial choice for industrial fleets based on distance-decay costs and expected CPUE.

### 2.2 Emergence
System-level tipping points (ecological stock collapse and social debt-trap cascades) emerge endogenously from interactions between exogenous climate shocks (MHWs) and fisher behavioral feedbacks under capacity subsidies.

### 2.3 Adaptation & Learning
- Fishers adapt spatial effort dynamically based on local stock depletion and fuel costs.
- Blue Carbon Managers adapt land-use allocations between mangrove preservation and shrimp aquaculture in response to carbon credit market prices and statutory MPA mandates.

### 2.4 Objectives
- SSF: Maximize household subsistence and net trip income while remaining debt-free.
- DWF: Maximize aggregate fleet profit across shelf and international waters.
- Blue Carbon Manager: Maximize combined land asset value ($NPV_{carbon} + NPV_{aquaculture}$).
- Trading Hub: Maximize trading margin subject to international trade frictions.

### 2.5 Interactions
- SSF and DWF compete directly for fish biomass across shelf boundary cells.
- Trading Hub acts as intermediary price-setter transmitting global commodity price shocks to local fishers.

---

## 3. Details

### 3.1 Initialization
- Environmental grid initialized with carrying capacity $K = 1,000 \text{ tonnes}$, distributed evenly across 100 cells at 70% virgin biomass ($B_0 = 700 \text{ tonnes}$).
- Initial SST, $pO_2$, and NPP set to year 2000 historical climatological baselines from CMIP6.
- 40 SSF agents and 15 DWF vessels initialized with heterogeneous working-capital debt $D_{i,0} = (0.15 + 0.72(1 - q_i))D_{\text{thresh},i}$ (`q_i = i/(N_{\text{ssf}}-1)`) and empirical baseline gear parameters over a 5,000 ha intertidal coastal zone.

### 3.2 Input Data & Ablation Switches
- Process-based Fish-MIP/CMIP6 shelf forcing parameterized from published `GFDL-ESM4` (`MOM6-COBALTv2`) and `MPI-ESM1-2-HR` (`MPIOM-HAMOCC6`) paths (SSP1-2.6, SSP2-4.5, SSP5-8.5).
- Self-contained empirical socio-economic and fisheries baseline parameters from GFW (satellite AIS & unreported offshore transshipment), UN Comtrade & FAO FishStatJ (bilateral trophic trade), OECD (subsidies), World Bank (poverty & blue natural capital), RAM Legacy v4.65 (stock health), and WTO ePing (NTMs) harmonized in `data/country_profiles.csv` (`data/raw/`).
- Causal mechanism ablation switches in `ABMEarthEngine` (`enable_debt_trap`, `enable_metabolic_compression`) isolate individual feedback pathways under `SSP5-8.5`.

### 3.3 Submodels
1. **ESM Climate Downscaling:** Updates SST, $pO_2$, and detects MHWs and compound shocks.
2. **Biophysical Stock Growth & Advection:** Implements logistic growth modulated by $\Phi$, carrying capacity modulated by NPP, and Laplacian bioclimatic diffusion.
3. **SSF Effort Allocation:** Evaluates candidate cells under Roy's Safety-First criterion.
4. **DWF Effort Allocation:** Softmax discrete choice model under subsidized fuel prices:
   $$C_{fuel} = D_{port} \cdot f_{dwf} \cdot P_{fuel} \cdot (1 - \sigma_{subsidy})$$
5. **Land-Use Optimization:** Computes marginal returns of mangrove blue carbon vs. aquaculture ponds.
6. **Market Clearing & Telecoupling:** Allocates seafood between domestic food security and high-margin exports based on tariff and NTM frictions.
7. **Policy Intervention:** Enforces MPA boundaries and transfers emergency safety net dividends.
