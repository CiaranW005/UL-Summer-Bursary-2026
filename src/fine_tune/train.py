from __future__ import annotations

import os
import json
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

from .evaluator import cluster_metrics

from .types import EmbeddingBatch, TrainingResults
from .transformer_block import DinoBlockExtension

from ..stats.mahalanobis_detector import MahalanobisDetector
from ..model_selection.types import EmbeddingPipeline

if TYPE_CHECKING:
    import torch.nn as nn

    from .logger import TrainLogger
    from .types import Sample, TrainingObjects, ModelInfo

def run_pipeline(
        views: torch.Tensor, 
        pipeline: EmbeddingPipeline
    ) -> torch.Tensor:
    """Pass views sequentially through a list of models."""

    x = views
    for stage in pipeline:
        x = stage.model(x)

    return x

def forward_model(
    x: torch.Tensor,
    pipeline: EmbeddingPipeline,
    model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embeds = run_pipeline(x, pipeline)

    if isinstance(model, DinoBlockExtension):
        z = model(x)
    else:
        z = model(embeds)

    return embeds, z

@contextmanager
def freeze_batchnorm_stats(
    model: torch.nn.Module
) -> Generator[None, None, None]:
    bn_layers = []
    states = []

    for module in model.modules():
        if isinstance(module, _BatchNorm):
            bn_layers.append(module)
            states.append(module.training)
            module.eval()

    try:
        yield
    finally:
        for module, state in zip(bn_layers, states):
            module.train(state)
     
class Trainer:
    def __init__(
            self,
            objs: TrainingObjects,
            pipeline: EmbeddingPipeline = EmbeddingPipeline(stages=[]),
            logger: TrainLogger | None = None,
        ) -> None:
        self.objs = objs
        self.pipeline = pipeline
        self.logger = logger

    def train_one_epoch(
            self,
            dataloader: DataLoader[Sample], 
        ) -> float:
        """Train the projection model for one epoch.

        Frozen inference models generate base embeddings for two augmented views.
        The trainable model projects those embeddings before the contrastive loss
        is calculated.

        Returns:
            The mean loss across all batches.
        """

        model = self.objs.model
        criterion = self.objs.criterion
        optimizer = self.objs.optimizer
        device = self.objs.device

        negatives = self.objs.negatives
        negative_labels = self.objs.negative_labels

        model.train()
        epoch_metrics: dict[str, float] = {}

        total_loss = 0.0
        for view1, view2, category, _ in dataloader:
            views = [view1, view2]

            batch_sizes = [v.shape[0] for v in views]
            x = torch.cat(views, dim=0).to(device) # Concatenated for batch_norm stats
            category = category.to(device)

            optimizer.zero_grad()

            embeds, z = forward_model(x, pipeline=self.pipeline, model=model)
            emb1, emb2 = torch.split(embeds, batch_sizes, dim=0)
            z1, z2 = torch.split(z, batch_sizes, dim=0)

            uses_batch_norm = getattr(model, "norm_type", None) == "batch"

            z_negs = negatives
            if z_negs is not None:
                if uses_batch_norm:
                    with freeze_batchnorm_stats(model):
                       z_negs = model(z_negs)
                else:
                    z_negs = model(z_negs)

            batch = EmbeddingBatch(
                org_view1=emb1,
                org_view2=emb2,
                proj_view1=z1,
                proj_view2=z2,
                negatives=z_negs,
                negative_labels=negative_labels,
                categories=category
            )

            # Contrastive based loss
            loss, metrics = criterion(batch)
            
            loss.backward()

            before = []
            grad_norm_sq = torch.tensor(0.0, device=device)
            for parameter in model.parameters():
                before.append(parameter.detach().clone())
                grad = parameter.grad

                if grad is None:
                    continue

                grad_norm_sq += grad.square().sum()
            grad_norm = grad_norm_sq.sqrt().item()

            optimizer.step()

            update_sq = torch.tensor(0.0, device=device)
            param_sq = torch.tensor(0.0, device=device)

            for old, new in zip(before, model.parameters()):
                update_sq += (new.detach() - old).square().sum()
                param_sq += old.square().sum()

            parameter_change =  (
                update_sq.sqrt()
                / param_sq.sqrt().clamp_min(1e-8)
            ).item()

            total_loss += loss.item()

            metrics["grad_norm"] = grad_norm
            metrics["parameter_change"] = parameter_change

            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

                if self.logger:
                    self.logger.log({f"batch/train/{key}": value})


        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Training dataloader produced no batches")

        if self.logger:
            self.logger.log({
                f"epoch/train/{key}": value / num_batches
                for key, value in epoch_metrics.items()
            })

        return total_loss / num_batches

    @torch.inference_mode()
    def evaluate(
            self,
            dataloader: DataLoader[Sample], 
            log_metrics: bool = False
        ) -> float:
        """Evaluate the projection model.

        Returns:
            The mean loss across all validation batches.
        """

        model = self.objs.model
        criterion = self.objs.criterion
        device = self.objs.device

        negatives = self.objs.negatives
        negative_labels = self.objs.negative_labels

        model.eval()
        epoch_metrics: dict[str, float] = {}

        total_loss = 0.0

        for view1, view2, category, _ in dataloader:
            views = [view1, view2]
            
            batch_sizes = [v.shape[0] for v in views]
            x = torch.cat(views, dim=0).to(device) # Concatenated for batch_norm stats
            category = category.to(device)

            embeds, z = forward_model(x, pipeline=self.pipeline, model=model)
            emb1, emb2 = torch.split(embeds, batch_sizes, dim=0)
            z1, z2 = torch.split(z, batch_sizes, dim=0)
            
            z_negs = negatives
            if z_negs is not None:
                z_negs = model(z_negs)
            
            batch = EmbeddingBatch(
                org_view1=emb1,
                org_view2=emb2,
                proj_view1=z1,
                proj_view2=z2,
                negatives=z_negs,
                negative_labels=negative_labels,
                categories=category
            )

            loss, metrics = criterion(batch)

            if criterion.collect_metrics and log_metrics:
                metrics["view_cosine"] = F.cosine_similarity(z1, z2, dim=-1).mean()

                clus_metric1 = cluster_metrics(z1, category)
                clus_metric2 = cluster_metrics(z2, category)

                metrics.update({
                    key: 0.5 * (clus_metric1[key] + clus_metric2[key])
                    for key in clus_metric1
                })

            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

                if self.logger:
                    self.logger.log({f"batch/eval/{key}": value})

            total_loss += loss.item()

        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Training dataloader produced no batches")

        if self.logger:
            self.logger.log({
                f"epoch/eval/{key}": value / num_batches
                for key, value in epoch_metrics.items()
            })
        
        return total_loss / num_batches

    @torch.inference_mode()
    def test(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        types_to_id: dict[str, int]
    ) -> pd.Series:
        metrics: dict[str, float] = {}

        train_embeds, train_categories, _ = self.embed_loader(train_loader)
        test_embeds, categories, types = self.embed_loader(test_loader)

        fitter = MahalanobisDetector(reg=1e-6)

        category_metrics = cluster_metrics(test_embeds, categories)

        metrics.update({
            f"test/category/{key}": value
            for key, value in category_metrics.items()
        })

        unique_cat = np.unique(categories)
        for cat in unique_cat:
            train_cat_mask = train_categories == cat
            test_cat_mask = categories == cat

            train_cat_embeds = train_embeds[train_cat_mask]
            test_cat_embeds = test_embeds[test_cat_mask]

            cat_types = types[test_cat_mask]

            # Individual defect type geometry
            type_metrics = cluster_metrics(test_cat_embeds, cat_types)

            metrics.update({
                f"test/{cat}/type/{key}": value
                for key, value in type_metrics.items()
            })

            # Normal vs defect geometry
            binary_labels = (
                cat_types != types_to_id["good"]
            ).astype(np.int64)

            normal_defect_metrics = cluster_metrics(test_cat_embeds, binary_labels)

            metrics.update({
                f"test/{cat}/normal_defect/{key}": value
                for key, value in normal_defect_metrics.items()
            })

            # OOD detection
            fitter.fit(train_cat_embeds)

            auc = fitter.evaluate_detection(
                good_embeds=test_cat_embeds[binary_labels == 0],
                defect_embeds=test_cat_embeds[binary_labels == 1]
            )

            metrics[f"test/{cat}/auroc"] = auc

        if self.logger:
            self.logger.log(metrics)

        return pd.Series(metrics)

    def train_model(
            self,
            *,
            epochs: int,
            train_loader: DataLoader,
            val_loader: DataLoader 
    ) -> TrainingResults:
        if epochs == 0:
            raise ValueError("Cannot train model for 0 epochs")
        
        history = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "gap", "time"])

        best_val = float("inf")
        best_epoch = None
        best_state = None

        val_window = deque(maxlen=5)

        pbar = tqdm(total=epochs, desc="Training")

        table_out = widgets.Output()
        display(table_out)

        try:
            for epoch in range(epochs):
                collect_val_metrics = epoch == 0 or (epoch + 1) % 5 == 0
                start = time.perf_counter()

                train_loss = self.train_one_epoch(train_loader)
                val_loss = self.evaluate(val_loader, log_metrics=collect_val_metrics)

                elapsed = time.perf_counter() - start

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch + 1

                    best_state = deepcopy(self.objs.model.state_dict())

                history.loc[len(history)] = {
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "gap": abs(val_loss - train_loss),
                    "time": f"{elapsed:.2f}"
                }

                with table_out:
                    table_out.clear_output(wait=True)
                    display(history.style.hide(axis="index"))
                pbar.update(1)

                val_window.append(val_loss)
                if len(val_window) == val_window.maxlen:
                    window = np.asarray(val_window)

                    x = np.arange(len(window))
                    slope = np.polyfit(x, window, 1)[0]

                    relative_slope = abs(slope) / abs(window.mean() + 1e-8)

                    if relative_slope < 1e-3:
                        _ = self.evaluate(val_loader, log_metrics=not collect_val_metrics)
                        break
                
        except Exception as e:
                return TrainingResults(
                    history=history,
                    best_state=best_state,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                    error=e
                )
        finally:
            pbar.close()

        return TrainingResults(
            history=history,
            best_state=best_state,
            best_val_loss=best_val,
            best_epoch=best_epoch
        )

    @torch.inference_mode()
    def embed_loader(
        self,
        dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.objs.model
        device = self.objs.device

        all_z = []
        all_categories = []
        all_types = []

        for view1, _, category, types in dataloader:
            view1 = view1.to(device)

            _, z = forward_model(view1, self.pipeline, model)

            all_z.append(z.cpu())
            all_categories.append(category)
            all_types.append(types)

        return (
            np.concatenate(all_z),
            np.concatenate(all_categories),
            np.concatenate(all_types)
        )

    @staticmethod
    def save_results(
            results: TrainingResults, 
            MODEL_DIR: Path,
            MODEL_INFO: ModelInfo,
            timestamp: str
        ):
        EXP_DIR = MODEL_DIR / timestamp

        os.makedirs(EXP_DIR, exist_ok=False)

        with open(EXP_DIR / "metadata.json", "w") as f:
            json.dump(MODEL_INFO, f, indent=4)

        results.history.to_csv(EXP_DIR / "train_history.csv", index=False)

        if results.best_state is not None:
            checkpoint = {
                "model_state_dict" :results.best_state,
                "parameters" : MODEL_INFO["parameters"],
                "parent_models" : MODEL_INFO["parent_models"]
            }

            torch.save(checkpoint, EXP_DIR / MODEL_INFO["parameters"]["model_name"])

            print(f"Best Model Saved (val_loss={results.best_val_loss:.4f}) as epoch {results.best_epoch}")
