# src/afterml.py
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)

def main():
    project_root = Path(__file__).parent.parent
    ckpt_dir = project_root / "horror_model" / "checkpoint-28392"

    # 1) Load tokenizer & model
    print("Loading tokenizer…")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    print(f"Loading model from {ckpt_dir}…")
    model = RobertaForSequenceClassification.from_pretrained(
        ckpt_dir,
        local_files_only=True
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to GPU:", torch.cuda.get_device_name(0))

    # 2) Read your reviews CSV
    csv_path = project_root / "data" / "phase1_labeled_reviews_optimized.csv"
    df = pd.read_csv(csv_path)
    df["reviews"] = df["reviews"].astype(str).fillna("")

    texts = df["reviews"].tolist()

    # 3) Batch inference
    batch_size = 32
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}

            out = model(**enc)
            probs = F.softmax(out.logits, dim=-1)
            batch_preds = torch.argmax(probs, dim=-1).cpu().tolist()
            preds.extend(batch_preds)

    # 4) Attach results
    df["predicted_label"] = preds
    # Map 0→–1, 1→+1 for consistency with your auto_sentiment scheme
    df["predicted_sentiment"] = df["predicted_label"].map({0: -1, 1: 1})

    # 5) Save to CSV
    out_csv = project_root / "horror_model" / "inference_results.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote inference results to:", out_csv)

if __name__ == "__main__":
    main()
