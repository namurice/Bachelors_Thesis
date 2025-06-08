# google colab

# Cell 0 – clean slate WITHOUT touching Triton
!pip uninstall -y jax jaxlib numpy torch torchvision torchaudio

# pin NumPy to a 1.x series
!pip install numpy==1.23.5

# install the official cu118 wheel (it brings back its own Triton bits)
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# pin a known‐good Transformers + deps
!pip install transformers==4.29.2 evaluate accelerate scikit-learn

!pip install --upgrade accelerate


!nvidia-smi

# Assuming you’ve uploaded your full CSV into /content/
!ls -lh /content/phase1_labeled_reviews_optimized.csv


import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Load + drop neutrals + map to 0/1
df = pd.read_csv("/content/phase1_labeled_reviews_optimized.csv")
df = df[df.auto_sentiment != 0].copy()
df["label"] = df.auto_sentiment.map({1: 1, -1: 0})

# Ensure 'reviews' column is string and handle potential NaNs
df["reviews"] = df["reviews"].astype(str).fillna("")

# 2) Stratified 90/10 split
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df["reviews"].tolist(),
    df["label"].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df["label"].tolist(),
)

print("Train:", len(train_texts), "Eval:", len(eval_texts))

import torch
from torch.utils.data import Dataset

class HorrorReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # ALWAYS returns only tensors
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# 1) Tokenizer + Datasets
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_ds  = HorrorReviewDataset(train_texts, train_labels, tokenizer)
eval_ds   = HorrorReviewDataset(eval_texts,  eval_labels,  tokenizer)

# 2) Model → GPU
print("CUDA OK?", torch.cuda.is_available(), torch.cuda.get_device_name(0))
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2
).cuda()

# 3) Trainer arguments
training_args = TrainingArguments(
    output_dir="/content/horror_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    fp16=True,
    report_to="none",       # disable W&B
)

# 4) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,     # <— note train_ds
    eval_dataset=eval_ds,       # <— note eval_ds
    data_collator=default_data_collator,
)


trainer.train()
# 1) Create a ZIP of the trained model folder
!zip -r /content/horror_model.zip /content/horror_model

# 2) Trigger a browser download (you’ll get a 'Save As' dialog—pick Desktop)
from google.colab import files
files.download("/content/horror_model.zip")


