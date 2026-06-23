import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report, f1_score
from datasets import Dataset

# ---------- Load data ----------
with open("data/dataset.json") as f:
    raw_data = json.load(f)

# Build label maps
all_labels = ["O"]
for item in raw_data:
    for ent in item["entities"]:
        attr = ent["label"]
        all_labels.append(f"B-{attr}")
        all_labels.append(f"I-{attr}")
all_labels = sorted(set(all_labels))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        return_offsets_mapping=True
    )
    labels = []
    for i in range(len(examples["text"])):
        offset_mapping = tokenized_inputs["offset_mapping"][i]
        entities = examples["entities"][i]
        aligned = ["O"] * len(offset_mapping)
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_start == offset_end == 0:
                    continue
                if offset_start >= start and offset_end <= end:
                    if offset_start == start:
                        aligned[idx] = f"B-{label}"
                    else:
                        aligned[idx] = f"I-{label}"
        label_ids = []
        for tag, (offset_start, offset_end) in zip(aligned, offset_mapping):
            if offset_start == 0 and offset_end == 0:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[tag])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    del tokenized_inputs["offset_mapping"]
    return tokenized_inputs

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["text", "entities"])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ---------- Collate function ----------
def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = [item["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(item["input_ids"])) for item in batch]
    attention_mask = [[1] * len(item["input_ids"]) + [0] * (max_len - len(item["input_ids"])) for item in batch]
    labels = [item["labels"] + [-100] * (max_len - len(item["labels"])) for item in batch]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# ---------- Optimizer & Scheduler ----------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ---------- Metrics ----------
def compute_metrics(preds, labels):
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    return {
        "f1": f1_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds, output_dict=True)
    }

# ---------- Training Loop ----------
best_f1 = 0
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds = []
    all_labels = []
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    f1 = metrics["f1"]
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained("./my_model")
        tokenizer.save_pretrained("./my_model")
        print("  -> Best model saved")

print(f"Training completed. Best F1: {best_f1:.4f}")
print("Model saved to ./my_model")