import torch

from tqdm import tqdm

def get_embeddings(model, loader, device):
    model.eval()

    embeddings = []
    indices = []

    with torch.no_grad():
        for images, idx in tqdm(loader):
            images = images.to(device)

            embs = model(images)

            embeddings.append(embs.cpu())
            indices.append(idx)
    
    embeddings = torch.cat(embeddings).numpy().astype("float32")
    indices = torch.cat(indices).numpy()
    
    return embeddings, indices