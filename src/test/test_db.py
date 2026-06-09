import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/sql/metadata.db")

print(pd.read_sql_query(
    "SELECT * FROM meta LIMIT 5",
    conn
))