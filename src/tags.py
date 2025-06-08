import pandas as pd
import numpy as np

# =============================================
# STEP 1: Load the cleaned CSV
# =============================================
try:
    df = pd.read_csv("game_descriptions.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("game_descriptions.csv", encoding='cp1252')

# =============================================
# STEP 2: Define fear-relevant features to keep
# =============================================
# Core tags tied to fear mechanics
TAGS_TO_KEEP = [
    # Gameplay
    "Survival Horror", "Psychological Horror", "First-Person", "Stealth", "No Combat",
    "Limited Resources", "Sanity Meter", "Jump Scares", "Quick-Time Events",
    # Themes
    "Atmospheric", "Gore", "Dark", "Isolation", "Claustrophobic Spaces",
    # Enemies
    "Zombies", "Supernatural", "Demons", "Animatronics", "Aliens", "Robots"
]

# Fear types and triggers
FEAR_TYPES = [
    "Body Horror", "Visceral Disgust", "Fear of the Unknown", "Existential Horror", 
    "Helplessness", "AI-Driven Paranoia", "Claustrophobia", "Jump Scares",
    "Existential Guilt", "Familial Guilt"
]

TRIGGERS_TO_KEEP = [
    "Las Plagas mutations", "gore", "unpredictable ghosts", "darkness",
    "voice recognition", "animatronic movements", "underwater isolation",
    "psychiatric horror", "Xenomorph's learning AI", "Pyramid Head"
]

# =============================================
# STEP 3: Process categorical features
# =============================================
# Clean and split comma-separated lists
df['Tags'] = df['Tags'].str.split(',')
df['Key Triggers'] = df['Key Triggers'].str.split(',')

# Create binary columns for selected features
for tag in TAGS_TO_KEEP:
    df[f"Tag_{tag.replace(' ', '_')}"] = df['Tags'].apply(
        lambda x: int(tag.strip() in [t.strip() for t in x]) if isinstance(x, list) else 0
    )

for fear in FEAR_TYPES:
    primary_match = df['primary fear type'].str.contains(fear, na=False).astype(int)
    secondary_match = df['Secondary Fear Type'].str.contains(fear, na=False).astype(int)
    df[f"Fear_{fear.replace(' ', '_')}"] = np.clip(primary_match + secondary_match, 0, 1)

for trigger in TRIGGERS_TO_KEEP:
    df[f"Trigger_{trigger.replace(' ', '_')}"] = df['Key Triggers'].apply(
        lambda x: int(trigger.strip() in [t.strip() for t in x]) if isinstance(x, list) else 0
    )

# =============================================
# STEP 4: Process numerical features
# =============================================
# Convert owner ranges to numerical midpoints
df['Estimated owners'] = df['Estimated owners'].apply(
    lambda x: np.mean([int(n) for n in str(x).split(' - ')]) if isinstance(x, str) else x
)

# Extract release year from date
df['Release Year'] = pd.to_datetime(df['Release date']).dt.year

# =============================================
# STEP 5: Add engineered features
# =============================================
# 1. Game length proxy (based on description word count)
df['Description_Length'] = df['About the game'].str.split().str.len()

# 2. Horror intensity (composite score)
horror_tags = ["Survival_Horror", "Psychological_Horror", "Gore", "Jump_Scares"]
df['Horror_Intensity'] = df[[f"Tag_{tag}" for tag in horror_tags]].sum(axis=1)

# =============================================
# STEP 6: Drop unused columns
# =============================================
columns_to_drop = [
    'Subgenres', 'Tags', 'primary fear type', 'Secondary Fear Type', 
    'Key Triggers', 'Release date', 'Game mechanics', 'About the game'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# =============================================
# STEP 7: Save cleaned data
# =============================================
df.to_csv("horror_games_processed.csv", index=False, encoding='utf-8')
print("Processed data saved to 'horror_games_processed.csv'")

# =============================================
# STEP 8: Verify output for Resident Evil 4
# =============================================
re4_features = df[df['Name'] == 'Resident Evil 4'].filter(regex='Tag_|Fear_|Trigger_')
print("\nResident Evil 4 - Active Fear Features:")
print(re4_features.loc[:, (re4_features != 0).any()].columns.tolist())