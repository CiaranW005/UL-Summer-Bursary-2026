import argparse

import torch
import pandas as pd

import subprocess
from pathlib import Path

from .utils import build_unsupervised_losses

from .run_experiment import run_experiment

from ...fine_tune.types import ModelParameters, ModelInfo

from ...config.loss_config import UNSUPERVISED_LOSS_CONFIGS
from ...config.model_configs import DinoFineTuneConfig
from ...config.paths import RESULTS

def main(pipeline_path: Path | None = None):
    model_configs = DinoFineTuneConfig()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    PARAMS: ModelParameters = {
        "seed": 42,
        "num_workers": 8,
        "pin_memory": torch.cuda.is_available(),
        
        "samples_per_category": 16,
        "batch_size": 15 * 16, # num categories * samples_per_category
        "epochs": 20,

        "model_dim": 384,
        "hidden_dim":  1536, # From dino documentation model_dim * mlp_factor, where mlp = 4
        "dropout": 0.0,
        "learning_rate": model_configs.learning_rates,
        "weight_decay": model_configs.weight_decays,
        "use_residual": False,
        
        "model_normaliser": "layer",
        "model_name": "model.pt",
    }

    NOTES = [
        "Trained on normal dino"
    ]

    results: list[pd.Series] = []
    for loss_config in UNSUPERVISED_LOSS_CONFIGS:
        loss_config.preservation_weight = 0
        
        MODEL_INFO: ModelInfo = {
            "model_type": "dino_fine_tune",
            "notes" : NOTES,
            "git_commit" : git_commit,
            "parameters" : PARAMS,
            "parent_models": {}
        }

        losses = build_unsupervised_losses(loss_config)

        metrics = run_experiment(
            params=PARAMS,
            pipeline_path=pipeline_path,
            losses=losses,
            model_info=MODEL_INFO
        )
        results.append(metrics)
    
    output_path = RESULTS / "dino_fine_tune" / "stage1_sweep.csv"
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






