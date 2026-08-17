#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Training with the MLP
PIPELINE_PATH="models/unsupervised_head/20260810_121201"

python -m src.experiments.training_sweeps.sweep_anomaly_head --path="$PIPELINE_PATH"
python -m src.experiments.training_sweeps.sweep_dino_adapter --path="$PIPELINE_PATH"