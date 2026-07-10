import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        embeds = model(batch)
        loss = criterion(embeds)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            embeds = model(batch)
            loss = criterion(embeds)

            total_loss += loss.item()

    return total_loss / len(dataloader)

