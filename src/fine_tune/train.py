import torch

from dataclasses import dataclass

@dataclass
class EmbeddingBatch:
    proj_view1: torch.Tensor
    proj_view2: torch.Tensor

    categories: torch.Tensor

    org_view1: torch.Tensor | None = None
    org_view2: torch.Tensor | None = None

    negatives : torch.Tensor | None = None

def run_models(x, models):
    for model in models:
        x = model(x)
    return x

def train_one_epoch(model, inf_models, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for view1, view2, category in dataloader:
        view1, view2 = view1.to(device), view2.to(device)
        category = category.to(device)

        optimizer.zero_grad()

        # Get embeddings from frozen dino
        with torch.no_grad():
            emb1 = run_models(view1, inf_models)
            emb2 = run_models(view2, inf_models)


        # Embeddings from the projection head 
        z1 = model(emb1)
        z2 = model(emb2)

        z_negs = model(criterion.negatives)

        batch = EmbeddingBatch(
            proj_view1=z1,
            proj_view2=z2,
            negatives=z_negs,
            categories=category
        )

        # Contrastive based loss
        loss, components = criterion(batch)
        
        loss.backward()

        grad_norm = 0.0

        for parameter in model.parameters():
            if parameter is not None:
                grad_norm += parameter.grad.norm().item()  ** 2

        grad_norm = grad_norm ** 0.5
        print("Gradient Norm:", grad_norm)

        before = next(model.parameters()).detach().clone()

        optimizer.step()

        after = next(model.parameters()).detach().clone()

        print("Paramter change:", (after - before).abs().mean().item())
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, inf_models, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for view1, view2, category in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            category = category.to(device)
            
            with torch.no_grad():
                emb1 = run_models(view1, inf_models)
                emb2 = run_models(view2, inf_models)

            z1 = model(emb1)
            z2 = model(emb2)

            z_negs = model(criterion.negatives)

            batch = EmbeddingBatch(
                proj_view1=z1,
                proj_view2=z2,
                negatives=z_negs,
                categories=category
            )

            loss, components = criterion(batch)

            total_loss += loss.item()

    return total_loss / len(dataloader)

