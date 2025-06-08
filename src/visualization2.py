import pandas as pd
import matplotlib.pyplot as plt
import os

# ── 1) Load your per‐game summary CSV ────────────────────────────────
csv_path = os.path.join("data", "game_summary.csv")
df = pd.read_csv(csv_path)

# ── 2) List the tag/fear columns you want to test ────────────────────
features = [
    "Tag_No_Combat", "Tag_Limited_Resources", "Tag_Sanity_Meter", "Tag_Jump_Scares",
    "Tag_Atmospheric", "Tag_Gore", "Tag_Isolation", "Tag_Zombies", "Tag_Supernatural",
    "Tag_Animatronics", "Tag_Aliens", "Tag_Robots", "Tag_Environmental_Storytelling",
    "Fear_Body_Horror", "Fear_Visceral_Disgust", "Fear_Fear_of_the_Unknown",
    "Fear_Powerlessness", "Fear_AI-Driven_Paranoia", "Fear_Claustrophobia",
    "Trigger_Unpredictable_Enemies", "Trigger_Psychiatric_Horror"
]

# ── 3) Compute each feature’s Pearson correlation with pred_avg ───────
corrs = df[features + ["pred_avg"]].corr()["pred_avg"].drop("pred_avg")
corrs = corrs.sort_values(ascending=True)

# ── 4) Plot a horizontal bar chart ───────────────────────────────────
plt.figure(figsize=(8, 6))
plt.barh(corrs.index, corrs.values, color="#2a9d8f", edgecolor="black")
plt.axvline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("Correlation with Fine-tuned Sentiment (pred_avg)")
plt.title("Which Game Tags & Fear Elements Predict Higher Review Scores?")
plt.tight_layout()
plt.savefig("feature_vs_pred_correlation.png", dpi=150)
plt.show()
