import os
import gc

import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW

from datetime import datetime
from pathlib import Path

from ...model_selection import load_inference_models

from ...config.paths import ROOT, DB_PATH, MODELS

from ...fine_tune import (
    get_device, load_dataset_meta,
    ModelInfo, ModelParameters, TrainingObjects,
    DataLoading, Trainer, TrainLogger,
    contrastive_transform, test_transform
)
from ...fine_tune.model import ProjectionHead
from ...fine_tune.transformer_block import DinoBlockExtension
from ...fine_tune.train import run_pipeline

from ...fine_tune.losses.combined_loss import CombinedLoss 
from ...fine_tune.losses.types import LossCollection

def run_experiment(
        params: ModelParameters,
        pipeline_path: Path | None,
        losses: LossCollection,
        model_info: ModelInfo
) -> pd.Series:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info["timestamp"] = timestamp

    model_type = model_info["model_type"]

    is_semi_supervised = model_type in {"anomaly_head", "dino_adapter_block"}

    MODEL_DIR = MODELS / model_type
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()

    pipeline = load_inference_models(pipeline_path, device=device)

    data = load_dataset_meta(DB_PATH)

    train_paths = data.train_paths
    test_paths = data.test_paths

    categories = data.categories
    types = data.types

    loader = DataLoading(root=ROOT, params=params, categories=categories, types=types)

    train_loader, val_loader = loader.create_train_val_loaders(
        train_paths,
        transform=contrastive_transform,
        train_ratio=0.95
    )

    test_dataset = loader.create_test_dataset(
        test_paths,
        transform=test_transform
    )

    neg_indices = np.empty(0, dtype=np.int64)
    if is_semi_supervised:
        neg_indices = loader.select_negative_indices(
            test_dataset,
            samples_per_category=5
        )
        model_info["negative_indices"] = neg_indices.tolist()

    test_loader = loader.create_test_loader(
        test_dataset,
        excluded_indices=neg_indices
    )

    neg_images, neg_labels = None, None
    if is_semi_supervised:
        neg_images, neg_labels = loader.get_neg_embeds(
            test_dataset,
            neg_indices,
            device
        )

    neg_embeds = neg_images
    if model_type == "dino_adapter_block":
        model = DinoBlockExtension(
            dino=pipeline.dino, 
            use_outer_residual=params["use_residual"], 
            dropout=params["dropout"]
        ).to(device)
    elif model_type == "dino_fine_tune":
        model = pipeline.dino
        pipeline.stages = []

        # params are already from loading as inference
        for p in model.blocks[-1].parameters():
            p.requires_grad = True

        for p in model.norm.parameters():
            p.requires_grad = True
    else:
        model = ProjectionHead(
            dim=params["model_dim"], hidden_dim=params["hidden_dim"], 
            norm_type=params["model_normaliser"],
            dropout=params["dropout"], residual=params["use_residual"]
        ).to(device)

        if neg_images is not None:
            neg_embeds = run_pipeline(neg_images, pipeline)
            
    model_info["parent_models"] = {
        stage.name: str(stage.path) if stage.path is not None else None
        for stage in pipeline
    }   

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), 
                      lr=params["learning_rate"], 
                      weight_decay=params["weight_decay"]
                    )

    logger = TrainLogger(project="MVTEC OOD", name=model_type, model_info=model_info)

    criterion = CombinedLoss(losses=losses)
    criterion.collect_metrics = logger is not None

    training_objs = TrainingObjects(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,

        negatives=neg_embeds,
        negative_labels=neg_labels
    )

    trainer = Trainer(objs=training_objs, pipeline=pipeline, logger=logger)

    try:
        results = trainer.train_model(
            epochs=params["epochs"], 
            train_loader=train_loader,
            val_loader=val_loader
        )
            
        trainer.save_results(
            results=results,
            MODEL_DIR=MODEL_DIR,
            MODEL_INFO=model_info,
            timestamp=timestamp
        )

        if results.error is not None:
            raise results.error

        test_metrics = trainer.test(
            train_loader=train_loader,
            test_loader=test_loader,
            types_to_id=loader.types_to_id

        )
    finally:
        del trainer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return test_metrics
