import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your per‐game summary
df = pd.read_csv("data/game_summary.csv")

# 2) Sort by Metacritic so the bars go top→bottom
df = df.sort_values("Metacritic score", ascending=True).reset_index(drop=True)

# 3) Scale Metacritic into [0,1] to match the [-1,1] sentiment range (we’ll clip to [0,1])
met_scaled = np.clip(df["Metacritic score"] / 100, 0, 1)

# 4) Build the horizontal grouped bar chart
y = np.arange(len(df))
height = 0.4

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(y - height/2, df["pred_avg"], height, label="Review Sentiment", color="#1f77b4")
ax.barh(y + height/2, met_scaled,    height, label="Metacritic (scaled)", color="#ff7f0e")

# 5) Labels & styling
ax.set_yticks(y)
ax.set_yticklabels(df["AppID"])
ax.set_xlabel("Scaled Value (0–1)")
ax.set_title("Review Sentiment vs. Metacritic (scaled) by Game")
ax.legend(loc="lower right")
ax.grid(axis="x", linestyle=":", linewidth=0.5)

plt.tight_layout()
plt.show()
