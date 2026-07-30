import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from collections.abc import Sequence

from .types import Sample, EmbeddingBatch

def run_models(
        x: torch.Tensor, 
        models: Sequence[nn.Module]
    ) -> torch.Tensor:
    """Pass a tensor sequentially through a list of models."""
    for model in models:
        x = model(x)
    return x

def train_one_epoch(
        model: nn.Module, 
        dataloader: DataLoader[Sample], 
        criterion: nn.Module, 
        optimizer: Optimizer, 
        device: torch.device,
        inf_models: list[nn.Module] | None = None
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

        # Get embeddings from frozen dino
        if inf_models is not None:
            with torch.no_grad():
                emb1 = run_models(view1, inf_models)
                emb2 = run_models(view2, inf_models)
        else:
            emb1 = view1
            emb2 = view2


        # Embeddings from the projection head 
        z1 = model(emb1)
        z2 = model(emb2)

        if hasattr(criterion, "negatives"):
            z_negs = model(criterion.negatives)
        else:
            z_negs = None

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

        grad_norm_sq: float = 0.0
        for parameter in model.parameters():
            grad = parameter.grad

            if grad is None:
                continue

            grad_norm_sq += float(torch.sum(grad * grad).item())

        # TODO: Log grad_norm
        grad_norm = grad_norm_sq**0.5

        before = next(model.parameters()).detach().clone()

        optimizer.step()

        after = next(model.parameters()).detach().clone()
        parameter_change = (after - before).abs().mean().item() # TODO: Log parameter_change

        #make a doct that returns data from steps here
        total_loss += loss.item()

    if len(dataloader) == 0:
        raise RuntimeError("Training dataloader produced no batches")

    return total_loss / len(dataloader)

def evaluate(
        model: nn.Module, 
        dataloader: DataLoader[Sample], 
        criterion: nn.Module, 
        device: torch.device,
        inf_models: list[nn.Module] | None = None
    ) -> float:
    """Evaluate the projection model.

    Returns:
        The mean loss across all validation batches.
    """
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for view1, view2, category, _ in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            category = category.to(device)

            if inf_models is not None:
                with torch.no_grad():
                    emb1 = run_models(view1, inf_models)
                    emb2 = run_models(view2, inf_models)
            else:
                emb1 = view1
                emb2 = view2

            z1 = model(emb1)
            z2 = model(emb2)

            if hasattr(criterion, "negatives"):
                z_negs = model(criterion.negatives)
            else:
                z_negs = None

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
