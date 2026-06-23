from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

ner = pipeline(
    "token-classification",
    model="./my_model",
    aggregation_strategy="simple"
)


class TextIn(BaseModel):
    text: str


def clean_entities(raw_entities: list[dict], original_text: str) -> dict[str, list[str]]:
    """
    Group entities by attribute and clean up common tokeniser issues.
    """
    result: dict[str, list[str]] = {}
    for ent in raw_entities:
        attr = ent["entity_group"]
        val = ent["word"].strip()
        # Fix hyphen spacing: "v - neck" -> "v-neck"
        val = val.replace(" - ", "-").replace(" -", "-").replace("- ", "-")

        if attr not in result:
            result[attr] = []
        result[attr].append(val)

    # --- Special case: incomplete "knee" length -----------------
    # If the model only extracted "knee" but the description actually
    # contains "knee length", extend it to the full phrase.
    if "length" in result:
        fixed_lengths = []
        for v in result["length"]:
            if v.lower() == "knee" and "knee length" in original_text.lower():
                fixed_lengths.append("knee length")
            else:
                fixed_lengths.append(v)
        result["length"] = fixed_lengths

    return result


@app.post("/extract")
def extract(payload: TextIn):
    raw_entities = ner(payload.text)
    cleaned = clean_entities(raw_entities, payload.text)
    return cleaned


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)