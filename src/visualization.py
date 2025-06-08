import pandas as pd
import matplotlib.pyplot as plt

# ── Load & prepare ─────────────────────────────────────────────────────────
df = pd.read_csv("data/game_summary.csv")
df["auto_avg"] = df["auto_avg"]
df["pred_avg"] = df["pred_avg"]

# map your AppIDs to human names

df["game"] = df["AppID"]

# ── Plot a colored dumbbell chart ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
ys = range(len(df))

# draw the connecting lines in light gray
for y, (_, row) in zip(ys, df.iterrows()):
    plt.plot(
        [row.auto_avg, row.pred_avg],
        [y, y],
        color="lightgray",
        linewidth=1,
    )

# draw auto-label points in blue
plt.scatter(
    df.auto_avg,
    ys,
    color="#1f77b4",   # matplotlib’s default “C0”
    label="Auto-label avg",
    zorder=3
)
# draw fine-tuned points in orange
plt.scatter(
    df.pred_avg,
    ys,
    color="#ff7f0e",   # matplotlib’s default “C1”
    label="Fine-tuned avg",
    zorder=3
)

plt.yticks(ys, df.game)
plt.xlabel("Mean sentiment (−1 to 1)")
plt.title("Auto-label vs. Fine-tuned Review Sentiment by Game")
plt.legend(loc="lower right")
plt.grid(axis="x", linestyle=":", alpha=0.7)
plt.xlim(-0.1, 1.0)
plt.tight_layout()
plt.savefig("game_sentiment_dumbbell.png", dpi=150)
plt.show()
