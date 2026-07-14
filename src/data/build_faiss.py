import faiss
import torch

from ..config.paths import FAISS_IDX, EMBEDS_DIR

def build_index(embs):
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(FAISS_IDX))

if __name__ == "__main__":
    embs = torch.load(EMBEDS_DIR / "raw_embeds.pt")

    build_index(embs=embs)