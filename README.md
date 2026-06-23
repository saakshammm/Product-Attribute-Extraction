# Product Attribute Extraction API

## Approach
- Fine-tuned DistilBERT for token classification (NER) on 200 synthetic product descriptions.
- 80/20 train-test split, trained for 10 epochs using PyTorch.
- Served via FastAPI endpoint `POST /extract`.

## Dataset
- 200 descriptions generated with random combinations of 8 attribute categories.
- Stored as character-level span annotations in JSON format.
- [dataset.json](https://github.com/saakshammm/Product-Attribute-Extraction/blob/main/data/dataset.json)

## Evaluation
- Token-level micro F1: 1.00 on the held-out test set.
- Manual inspection revealed minor tokenizer artefacts (hyphen spacing, incomplete multi-word spans), which were resolved with post-processing.
- Attribute-level exact-match accuracy: 100% on the synthetic test set; real-world descriptions may differ (see failure cases).

## Failure Cases
- Unseen phrasing (e.g., "falls to the knee" instead of "knee length") — model fails to extract length.
- Hyphenated attributes like "v-neck" get split by the tokenizer; fixed via post-processing.
- Multiple values for the same attribute (e.g., two colors) may be merged or missed.

## How to Run
```bash
pip install -r requirements.txt
python app.py
````

API available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Endpoint

**POST /extract**

### Input

```json
{
  "text": "Elegant black dress crafted from cotton fabric featuring a v-neck, short sleeve, knee length and none detailing."
}
```

### Output

```json
{
  "color": ["black"],
  "category": ["dress"],
  "fabric": ["cotton"],
  "neckline": ["v-neck"],
  "sleeve": ["short sleeve"],
  "length": ["knee length"],
  "silhouette": ["a-line"]
}
```
