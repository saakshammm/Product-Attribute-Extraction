import pandas as pd
import json
import re

df = pd.read_csv("data/dataset.csv")

attribute_cols = [
    "silhouette", "fabric", "neckline", "sleeve",
    "length", "embellishment", "color", "category"
]

records = []

for _, row in df.iterrows():
    text = row["description"]
    entities = []
    for attr in attribute_cols:
        raw_value = row[attr]
        if pd.isna(raw_value) or str(raw_value).strip().lower() in ["none", ""]:
            continue
        value = str(raw_value).strip()
        match = re.search(re.escape(value), text, re.IGNORECASE)
        if match:
            start, end = match.start(), match.end()
            entities.append({"start": start, "end": end, "label": attr})
        else:
            match = re.search(re.escape(value.lower()), text.lower())
            if match:
                start, end = match.start(), match.end()
                entities.append({"start": start, "end": end, "label": attr})
            else:
                print(f"Warning: '{value}' not found in:\n{text[:80]}...\n")
    records.append({"text": text, "entities": entities})

with open("data/dataset.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Converted {len(records)} descriptions to dataset.json")