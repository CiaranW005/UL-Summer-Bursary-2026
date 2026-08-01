import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .types import Sample, EmbeddingBatch
from ..model_selection import EmbeddingPipeline

def run_pipeline(
        views: list[torch.Tensor], 
        pipeline: EmbeddingPipeline
    ) -> tuple[torch.Tensor, ...]:
    """Pass views sequentially through a list of models."""
    if not views:
        return ()

    batch_sizes = [v.shape[0] for v in views]
    x = torch.cat(views, dim=0)

    for stage in pipeline:
        x = stage.model(x)

    return torch.split(x, batch_sizes, dim=0)

def train_one_epoch(
        model: nn.Module, 
        dataloader: DataLoader[Sample], 
        criterion: nn.Module, 
        optimizer: Optimizer, 
        device: torch.device,
        pipeline: EmbeddingPipeline = EmbeddingPipeline(stages=[])
    ) -> float:
    """Train the projection model for one epoch.

    Frozen inference models generate base embeddings for two augmented views.
    The trainable model projects those embeddings before the contrastive loss
    is calculated.

    Returns:
        The mean loss across all batches.
    """
    model.train()

    total_loss = 0.0
    for batch_idx, (view1, view2, category, _) in enumerate(dataloader): # TODO: Log the batch idx info comes from
        view1, view2 = view1.to(device), view2.to(device)
        category = category.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb1, emb2 = run_pipeline([view1, view2], pipeline)
  
        # Embeddings from the projection head 
        z1 = model(emb1)
        z2 = model(emb2)

        z_negs = model(criterion.negatives) if hasattr(criterion, "negatives") else None

        batch = EmbeddingBatch(
            org_view1=emb1,
            org_view2=emb2,
            proj_view1=z1,
            proj_view2=z2,
            negatives=z_negs,
            categories=category
        )

        # Contrastive based loss
        loss, _ = criterion(batch)
        
        loss.backward()

        grad_norm_sq = torch.tensor(0.0, device=device)
        for parameter in model.parameters():
            grad = parameter.grad

            if grad is None:
                continue

            grad_norm_sq += grad.square().sum()

        # TODO: Log grad_norm
        grad_norm = grad_norm_sq.sqrt().item()

        before = next(model.parameters()).detach().clone()

        optimizer.step()

        after = next(model.parameters()).detach().clone()
        parameter_change = (after - before).abs().mean().item() # TODO: Log parameter_change

        #make a doct that returns data from steps here
        total_loss += loss.item()

    if len(dataloader) == 0:
        raise RuntimeError("Training dataloader produced no batches")

    return total_loss / len(dataloader)

@torch.inference_mode()
def evaluate(
        model: nn.Module, 
        dataloader: DataLoader[Sample], 
        criterion: nn.Module, 
        device: torch.device,
        pipeline: EmbeddingPipeline = EmbeddingPipeline(stages=[])
    ) -> float:
    """Evaluate the projection model.

    Returns:
        The mean loss across all validation batches.
    """
    model.eval()

    total_loss = 0.0

    for view1, view2, category, _ in dataloader:
        view1, view2 = view1.to(device), view2.to(device)
        category = category.to(device)

        emb1, emb2 = run_pipeline([view1, view2], pipeline)

        z1 = model(emb1)
        z2 = model(emb2)

        z_negs = model(criterion.negatives) if hasattr(criterion, "negatives") else None

        batch = EmbeddingBatch(
            org_view1=emb1,
            org_view2=emb2,
            proj_view1=z1,
            proj_view2=z2,
            negatives=z_negs,
            categories=category
        )

        loss, _ = criterion(batch)

        total_loss += loss.item()

    if len(dataloader) == 0:
            raise RuntimeError("Evaluation dataloader produced no batches.")
    
    return total_loss / len(dataloader)
