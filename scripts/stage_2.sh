#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Training with fine tuned dino
PIPELINE_PATH="models/dino_fine_tune/20260809_161959"

python -m src.experiments.training_sweeps.sweep_unsupervised_head --path="$PIPELINE_PATH"
python -m src.experiments.training_sweeps.sweep_anomaly_head --path="$PIPELINE_PATH"
python -m src.experiments.training_sweeps.sweep_dino_adapter --path="$PIPELINE_PATH"

# Training with the MLP
PIPELINE_PATH="models/unsupervised_head/20260809_221306"

python -m src.experiments.training_sweeps.sweep_anomaly_head --path="$PIPELINE_PATH"
python -m src.experiments.training_sweeps.sweep_dino_adapter --path="$PIPELINE_PATH"