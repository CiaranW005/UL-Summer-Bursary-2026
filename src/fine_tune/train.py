import torch

def train_one_epoch(model, dino, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for view1, view2 in dataloader:
        view1, view2 = view1.to(device), view2.to(device)

        optimizer.zero_grad()

        # Get embeddings from frozen dino
        with torch.no_grad():
            emb1 = dino(view1)
            emb2 = dino(view2)


        # Embeddings from the projection head 
        z1 = model(emb1)
        z2 = model(emb2)

        # Contrastive based loss
        loss = criterion(z1, z2)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dino, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for view1, view2 in dataloader:
            view1, view2 = view1.to(device), view2.to(device)
            
            with torch.no_grad():
                emb1 = dino(view1)
                emb2 = dino(view2)

            z1 = model(emb1)
            z2 = model(emb2)

            loss = criterion(z1, z2)

            total_loss += loss.item()

    return total_loss / len(dataloader)

