from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "mvtec_ad"
SQL_DIR = DATA_DIR / "sql"
CACHES = DATA_DIR / "cache"

DB_PATH = SQL_DIR / "metadata.db"
CSV_PATH = DATA_DIR / "metadata.csv"

EXPERIMENTS = DATA_DIR / "experiments" 
RESULTS = DATA_DIR / "results"
EMBEDS_DIR = DATA_DIR / "embeddings"

FAISS_DIR = DATA_DIR / "faiss"
FAISS_IDX = FAISS_DIR / "embeds.index"

MODEL_DIR = ROOT / "models"

IMAGES = ROOT / "images"