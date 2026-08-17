
import json

from pathlib import Path

from typing import cast

from .types import EmbeddingPipeline, PipelineStage

from ..config.paths import EMBEDS_DIR

def create_pipeline(model_dir: Path) -> EmbeddingPipeline:
    model_metadata = model_dir / "metadata.json"

    with open(model_metadata, "r") as f:
        meta = json.load(f)

    parent_models = cast(dict[str, str], meta["parent_models"])
    model_type = meta.get("model_type")

    stages: list[PipelineStage] = []
    for name, path in parent_models.items():
        model_path = None
        if path in ("None", None):
            embed_path = EMBEDS_DIR / "dino/pretrained"
        else:
            path = Path(path)
            if (
                "dino_adapter_block" in path.parts
                or "dino_fine_tune" in path.parts
            ):
                parent_name = "dino"
            else:
                parent_name = path.parent.parent.name
            embed_path = EMBEDS_DIR / parent_name / path.parent.name
            model_path = path.parent
        
        stages.append(PipelineStage(name=name, model_path=model_path, embed_path=embed_path))

    if (
        "dino_adapter_block" in model_dir.parts
        or "dino_fine_tune" in model_dir.parts
    ):
        parent_name = "dino"
    else:
        parent_name = model_dir.parent.name
    stages.append(PipelineStage(
        name=model_type or "chosen_model",
        model_path=model_dir,
        embed_path=EMBEDS_DIR / parent_name / model_dir.name
    ))
    return EmbeddingPipeline(stages=stages)



