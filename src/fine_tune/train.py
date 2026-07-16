import torch

from dataclasses import dataclass

@dataclass
class EmbeddingBatch:
    org_view1: torch.Tensor
    org_view2: torch.Tensor

    proj_view1: torch.Tensor
    proj_view2: torch.Tensor

    categories: torch.Tensor

def train_one_epoch(model, dino, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for view1, view2, category in dataloader:
        view1, view2 = view1.to(device), view2.to(device)
        category = category.to(device)

        optimizer.zero_grad()

        # Get embeddings from frozen dino
        with torch.no_grad():
            emb1 = dino(view1)
            emb2 = dino(view2)


        # Embeddings from the projection head 
        z1 = model(emb1)
        z2 = model(emb2)

        batch = EmbeddingBatch(
            org_view1=emb1,
            org_view2=emb2,
            proj_view1=z1,
            proj_view2=z2,
            categories=category
        )

        # Contrastive based loss
        loss, components = criterion(batch)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dino, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for view1, view2, category in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            category = category.to(device)
            
            with torch.no_grad():
                emb1 = dino(view1)
                emb2 = dino(view2)

            z1 = model(emb1)
            z2 = model(emb2)

            batch = EmbeddingBatch(
                org_view1=emb1,
                org_view2=emb2,
                proj_view1=z1,
                proj_view2=z2,
                categories=category
            )

            loss, components = criterion(batch)

            total_loss += loss.item()

    return total_loss / len(dataloader)

