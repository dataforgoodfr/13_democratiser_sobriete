import os
import pandas as pd
from tqdm import tqdm

d = {}
for filename in tqdm(os.listdir("txt")):
    oaid = filename.split(".")[0]
    with open(os.path.join("txt", filename), "r", encoding="utf-8") as f:
        text = f.read()
    d[oaid] = text

df = pd.DataFrame.from_dict(d, orient="index", columns=["text"])
df.index.name = "oa_id"
df["length"] = df["text"].str.len()
df.to_parquet("raw_texts.parquet")
