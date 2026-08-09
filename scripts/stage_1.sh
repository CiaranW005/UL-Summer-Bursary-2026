#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m src.experiments.training_sweeps.sweep_dino_fine_tune

python -m src.experiments.training_sweeps.sweep_unsupervised_head

python -m src.experiments.training_sweeps.sweep_anomaly_head

python -m src.experiments.training_sweeps.sweep_dino_adapter