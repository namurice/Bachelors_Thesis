import re
import numpy as np
import pandas as pd
import spacy
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# ── 0) Adjust these to match your project structure ───────────────────────
ROOT = Path(__file__).parent.parent
INPUT_CSV  = ROOT / "data" / "initial data" / "resident_evil_reviews.csv"
OUTPUT_CSV = ROOT / "data" / "resident_evil_auto.csv"

# ── 1) Load data ──────────────────────────────────────────────────────────
print(f"Loading reviews from {INPUT_CSV}…")
df = pd.read_csv(INPUT_CSV, encoding="latin1", on_bad_lines="skip")

# Clean only the 'reviews' column of any bad‐UTF8 sequences
df["reviews"] = df["reviews"].apply(
    lambda x: x.encode("utf-8", "ignore").decode("utf-8")
              if isinstance(x, str) else x
)

# ── 2) Prep for batching ───────────────────────────────────────────────────
batch_size    = 1000
total_reviews = len(df)
results       = np.empty(total_reviews, dtype=int)

# ── 3) Load NLP tools once ────────────────────────────────────────────────
print("Loading spaCy model & VADER…")
nlp      = spacy.load("en_core_web_lg", disable=["parser", "ner"])
analyzer = SentimentIntensityAnalyzer()

intensity_pattern = re.compile(
    r"\b(a{4,}|e{4,}|fuck|shit|brrrr|omfg|holy)\b",
    re.IGNORECASE
)

HORROR_WORDS = [nlp(w)[0] for w in [
    "scary", "terrifying", "eerie", "horrifying", "frightening",
    "spooky", "chilling", "haunting", "petrifying", "bloodcurdling",
    "peed", "shat", "intense", "scream", "creepy", "disturbing"
]]
POSITIVE_WORDS = [nlp(w)[0] for w in [
    "fun", "love", "amazing", "best", "great", "awesome",
    "enjoy", "perfect", "fantastic", "recommend", "excellent", "wonderful"
]]

def is_intense(text: str) -> bool:
    return bool(intensity_pattern.search(text)) if isinstance(text, str) else False

def is_horror_positive(doc) -> bool:
    # any horror word + an un‐negated positive?
    horror_present = any(
        token.similarity(hw) > 0.65
        for token in doc
        for hw in HORROR_WORDS
    )
    if not horror_present:
        return False
    for token in doc:
        if any(token.similarity(pw) > 0.65 for pw in POSITIVE_WORDS):
            # check for a negation within 3 tokens before
            if not any(
                child.dep_ == "neg"
                and child.i in range(token.i - 3, token.i + 1)
                for child in token.children
            ):
                return True
    return False

def process_batch(texts):
    batch_scores = []
    docs = list(nlp.pipe(texts))
    for doc in docs:
        txt = doc.text.strip()
        if not txt:
            batch_scores.append(-1)
            continue

        score = analyzer.polarity_scores(txt)["compound"]
        if is_horror_positive(doc):
            score += max(0.2, min(0.6, 0.5 - score))
        if is_intense(txt) and score > -0.5:
            score += 0.3

        # map to -1, 0, +1
        label =  1 if score >  0.3 else \
                -1 if score < -0.3 else 0
        batch_scores.append(label)

    return batch_scores

# ── 4) Main processing loop ───────────────────────────────────────────────
print(f"Processing {total_reviews} reviews in batches of {batch_size}…")
for start in tqdm(range(0, total_reviews, batch_size)):
    end = min(start + batch_size, total_reviews)
    batch = df["reviews"].iloc[start:end].fillna("").tolist()
    results[start:end] = process_batch(batch)

# ── 5) Save results ───────────────────────────────────────────────────────
df["auto_sentiment"] = results
print("\nLabel distribution:\n", df["auto_sentiment"].value_counts(), "\n")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"✅ Auto-labeled CSV saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    # executed when you run `python src/autolabel.py`
    pass
