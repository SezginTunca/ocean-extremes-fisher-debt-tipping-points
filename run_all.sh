#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== ABM-Earth-TropEcS Core Scientific Replication Pipeline ==="
python3 model/calibration.py
python3 analysis/surrogate_emulator.py
python3 analysis/global_sensitivity_analysis.py
python3 analysis/compare_esm_models.py
python3 analysis/econometric_analysis.py
python3 analysis/tipping_point_detector.py
python3 model/stochastic_evaluator.py
python3 analysis/mechanism_ablation.py
python3 visualizations/generate_figures.py
python3 visualizations/generate_supplementary_figures.py
echo "=== Core scientific replication pipeline completed ==="
