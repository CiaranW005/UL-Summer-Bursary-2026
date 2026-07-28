from dataclasses import dataclass
from pathlib import Path

from .metadata import get_parent_dino

@dataclass(frozen=True)
class ModelSelection:
    dino_path: Path | None
    category_head_path: Path | None

def resolve_anomaly_training_selection(
    use_category_head: bool,
    selected_dino: Path | None,
    selected_category_head: Path | None,
) -> ModelSelection:
    if use_category_head:
        if selected_category_head is None:
            raise ValueError(
                "A category head must be selected"
            )

        dino_path = get_parent_dino(
            selected_category_head
        )

        return ModelSelection(
            dino_path=dino_path,
            category_head_path=selected_category_head,
        )

    return ModelSelection(
        dino_path=selected_dino,
        category_head_path=None,
    )