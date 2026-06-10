from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "mvtec_ad"
SQL_DIR = DATA_DIR / "sql"
DB_PATH = SQL_DIR / "metadata.db"
CSV_PATH = DATA_DIR / "metadata.csv"

EMBEDS_DIR = DATA_DIR / "embeddings"

FAISS_DIR = DATA_DIR / "faiss"
FAISS_IDX = FAISS_DIR / "embeds.index"

MODEL_DIR = ROOT / "models"