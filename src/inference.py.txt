# colab

# 1) (Re)install a compatible NumPy
!pip install --upgrade numpy==1.23.5

# 2) Install PyTorch (with CUDA support if you’re on GPU) and a compatible torchvision
# We explicitly install torchvision 0.16.2 to match PyTorch 2.2.2 compiled with cu118
!pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# 3) Install Transformers
!pip install transformers


from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, pipeline
import torch

MODEL_DIR = "/content/horror_model/content/horror_model/checkpoint-28392"   # ← adjust if your zip landed somewhere else

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model     = RobertaForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).cuda()

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,           # GPU
)

# quick smoke test
print(clf([
    "I was terrified but I loved it!",
    "This game sucks and scared me off."
]))



import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# ── 1) Load your auto-labeled CSV ───────────────────────────────────────
# (Replace this path with wherever you uploaded it in Colab)
INPUT_CSV  = "/content/resident_evil_auto.csv"
OUTPUT_CSV = "/content/resident_evil_final.csv"

df = pd.read_csv(INPUT_CSV)
# if you want to drop pure-neutrals:
df = df[df.auto_sentiment != 0].copy()
texts = df["reviews"].astype(str).tolist()

# ── 2) Re-load tokenizer & model ───────────────────────────────────────
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_DIR,      # same MODEL_DIR as above
    local_files_only=True
).eval().cuda()

# ── 3) Inference loop with tqdm ────────────────────────────────────────
all_labels, all_scores = [], []
for i in tqdm(range(0, len(texts), 64), desc="Inferring"):
    batch = texts[i : i + 64]
    enc   = tokenizer(
        batch,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)

    all_labels.extend(probs.argmax(-1).cpu().tolist())
    all_scores.extend(probs.max(-1).values.cpu().tolist())

# ── 4) Attach predictions & save ───────────────────────────────────────
# map 1→+1, 0→–1 if you prefer
df["pred_sentiment"] = [1 if l==1 else -1 for l in all_labels]
df["pred_score"]     = all_scores

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Scored CSV saved to {OUTPUT_CSV}")
