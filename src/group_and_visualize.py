import os
import pandas as pd

def main():
    here = os.path.dirname(__file__)
    data_in  = os.path.join(here, "..", "data", "resident_evil_final.csv")
    data_out = os.path.join(here, "..", "data", "resident.csv")

    # 1) Load everything
    df = pd.read_csv(data_in)

    # 2) Compute per‐game averages + counts
    grouped = (
        df
        .groupby("AppID")
        .agg(
            auto_avg      = ("auto_sentiment",  "mean"),
            pred_avg      = ("pred_sentiment",  "mean"),
            pred_score_avg= ("pred_score",      "mean"),  # if you want it
            count         = ("reviews",         "size"),
        )
        .reset_index()
    )

    # 3) Grab your game‐level metadata columns
    #    (all columns except the per‐review ones)
    meta_cols = [
        "AppID",
        "official_description_sentiment_score",
        "Metacritic score",
        "Tag_No_Combat", "Tag_Limited_Resources", "Tag_Sanity_Meter",
        "Tag_Jump_Scares", "Tag_Atmospheric", "Tag_Gore",
        "Tag_Isolation", "Tag_Zombies", "Tag_Supernatural",
        "Tag_Animatronics", "Tag_Aliens", "Tag_Robots",
        "Tag_Environmental_Storytelling",
        "Fear_Body_Horror", "Fear_Visceral_Disgust", "Fear_Fear_of_the_Unknown",
        "Fear_Powerlessness", "Fear_AI-Driven_Paranoia", "Fear_Claustrophobia",
        "Trigger_Unpredictable_Enemies", "Trigger_Psychiatric_Horror",
        "Total_Horror_Elements",
        # (and any other game‐level columns you want to keep)
    ]
    meta = df[meta_cols].drop_duplicates(subset="AppID")

    # 4) Merge metadata + your new stats
    summary = pd.merge(meta, grouped, on="AppID")

    # 5) Save just those 11 rows × (meta + auto_avg, pred_avg, count, ...)
    summary.to_csv(data_out, index=False)
    print(f"✅ Wrote game summary to {data_out}")

if __name__=="__main__":
    main()
