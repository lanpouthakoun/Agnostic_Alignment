import json
import pyarrow as pa
import pyarrow.parquet as pq

input_path = "dataset.jsonl"
output_path = "dataset.parquet"

rows = []

print("Loading dataset.jsonl ...")
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

print(f"Loaded {len(rows)} rows. Converting to Parquet...")

table = pa.Table.from_pylist(rows)

pq.write_table(table, output_path)

print(f"Saved Parquet file to {output_path}")
