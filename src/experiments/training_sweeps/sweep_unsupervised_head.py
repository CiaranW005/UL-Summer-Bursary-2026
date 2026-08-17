import argparse

import torch
import pandas as pd

import subprocess
from pathlib import Path

from .utils import build_unsupervised_losses

from .run_experiment import run_experiment

from ...fine_tune.types import ModelParameters, ModelInfo

from ...config.loss_config import UNSUPERVISED_LOSS_CONFIGS
from ...config.model_configs import MLPConfig
from ...config.paths import RESULTS

def main(pipeline_path: Path | None):
    model_configs = MLPConfig()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    PARAMS: ModelParameters = {
        "seed": 42,
        "num_workers": 4,
        "pin_memory": torch.cuda.is_available(),
        
        "samples_per_category": 32,
        "batch_size": 15 * 32, # num categories * samples_per_category
        "epochs": 20,

        "model_dim": 384,
        "hidden_dim":  1536, # From dino documentation model_dim * mlp_factor, where mlp = 4
        "dropout": model_configs.dropout,
        "learning_rate": model_configs.learning_rate,
        "weight_decay": model_configs.weight_decay,
        "use_residual": model_configs.residual,
        
        "model_normaliser": "layer",
        "model_name": "model.pt",
    }

    NOTES = [
        "Trained on normal dino"
    ]

    results: list[pd.Series] = []
    for loss_config in UNSUPERVISED_LOSS_CONFIGS:
        losses = build_unsupervised_losses(loss_config)

        MODEL_INFO: ModelInfo = {
            "losses": losses.to_dict(),
            "model_type": "unsupervised_head",
            "notes" : NOTES,
            "git_commit" : git_commit,
            "parameters" : PARAMS,
            "parent_models": {}
        }

        metrics = run_experiment(
            params=PARAMS,
            pipeline_path=pipeline_path,
            losses=losses,
            model_info=MODEL_INFO
        )
        results.append(metrics)
    
    output_path = RESULTS / "unsupervised_head" / "stage1_sweep.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=Path,
        required=False,
        help="Path to the embedding/model pipeline(No path means DINO)",
        default=None
    )

    args = parser.parse_args()

    main(args.path)






