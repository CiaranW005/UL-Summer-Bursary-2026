from pathlib import Path
from typing import Any

import torch

from src.config.paths import MODELS


def load_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )


def get_model_info(
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)

    return checkpoint.get("model_info", {})


def get_parent_models(
    checkpoint_path: Path,
) -> dict[str, str | None]:
    model_info = get_model_info(checkpoint_path)

    return model_info.get(
        "parent_models",
        {},
    )


def resolve_model_path(
    stored_path: str | None,
) -> Path | None:
    if stored_path is None:
        return None

    if stored_path == "pretrained:dinov2_vits14":
        return None

    path = Path(stored_path)

    if not path.is_absolute():
        path = MODELS / path

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {path}"
        )

    return path


def get_parent_dino(
    checkpoint_path: Path,
) -> Path | None:
    parents = get_parent_models(checkpoint_path)

    return resolve_model_path(
        parents.get("dino")
    )


def get_parent_category_head(
    checkpoint_path: Path,
) -> Path | None:
    parents = get_parent_models(checkpoint_path)

    return resolve_model_path(
        parents.get("category_head")
    )