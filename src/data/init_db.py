from pathlib import Path
import sqlite3
import pandas as pd

DATA_DIR = Path("../data")
DB_DIR = DATA_DIR / "sql"
DB_PATH = DB_DIR / "metadata.db"
CSV_PATH = DATA_DIR / "metadata.csv"

def build_metadata_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Ensure FAISS index id == dataframe row id == SQL id
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    conn = sqlite3.connect(DB_PATH)

    df.to_sql(
        "meta",
        conn,
        if_exists="replace",
        index=False
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_category ON meta(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_split ON meta(split)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_label ON meta(label)")

    conn.commit()
    conn.close()

    print(f"Created database at: {DB_PATH}")
    print(f"Rows inserted: {len(df)}")

if __name__ == "__main__":
    build_metadata_db()