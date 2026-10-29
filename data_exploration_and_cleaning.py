import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "train.csv"
valid_path = DATA_DIR / "valid.csv"
test_path = DATA_DIR / "test.csv"

# Fallback: if files are not present locally, read directly from Hugging Face
train_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/last_out/Video_Games.train.csv"
valid_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/last_out/Video_Games.valid.csv"
test_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/0core/last_out/Video_Games.test.csv"

print("--- Step 1: Loading Interaction Datasets ---")
train_df = pd.read_csv(train_path if train_path.exists() else train_url)
valid_df = pd.read_csv(valid_path if valid_path.exists() else valid_url)
test_df = pd.read_csv(test_path if test_path.exists() else test_url)

print(f"Raw Train Shape: {train_df.shape}")
print(f"Raw Valid Shape: {valid_df.shape}")
print(f"Raw Test Shape:  {test_df.shape}")

# ---------------------------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA) on Train Split
# ---------------------------------------------------------------------------
print("\n--- Step 2: Exploratory Data Analysis (EDA) ---")
total_interactions = len(train_df)
num_users = train_df["user_id"].nunique()
num_games = train_df["parent_asin"].nunique()

# 1. Sparsity check
matrix_sparsity = (1.0 - (total_interactions / (num_users * num_games))) * 100
print(f"Unique Users: {num_users:,}")
print(f"Unique Games: {num_games:,}")
print(f"Matrix Sparsity: {matrix_sparsity:.6f}%")

# 2. Rating imbalance analysis
print("\nRating Distribution in Train:")
rating_counts = train_df["rating"].value_counts().sort_index()
print(rating_counts)
high_rating_ratio = (train_df["rating"] >= 4.0).mean() * 100
print(f"Imbalance Ratio (Rating >= 4.0): {high_rating_ratio:.2f}% of total data")

# 3. User activity analysis
user_counts = train_df.groupby("user_id")["parent_asin"].count()
print(f"Mean interactions per user: {user_counts.mean():.2f}")
print(f"Median interactions per user: {user_counts.median():.0f}")
print(f"Users with only 1 rating: {(user_counts == 1).sum():,} ({(user_counts == 1).sum() / num_users * 100:.2f}%)")

# Save EDA visualization plot
plt.figure(figsize=(7, 4))
plt.bar(rating_counts.index, rating_counts.values, color="steelblue")
plt.title("Distribution of Video Game Ratings")
plt.xlabel("Rating (Stars)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(DATA_DIR / "eda_rating_distribution.png")
plt.close()
print("Saved plot to: dataset/eda_rating_distribution.png")

# ---------------------------------------------------------------------------
# Step 3: Cleaning & Preprocessing (Missing values, Duplicates, Mappings)
# ---------------------------------------------------------------------------
print("\n--- Step 3: Cleaning & Index Mapping ---")

# 1. Drop missing critical fields
train_df = train_df.dropna(subset=["user_id", "parent_asin", "rating"])
valid_df = valid_df.dropna(subset=["user_id", "parent_asin", "rating"])
test_df = test_df.dropna(subset=["user_id", "parent_asin", "rating"])

# 2. Deduplication: keep the latest interaction for the same user-game pair
if "timestamp" in train_df.columns:
    train_df = train_df.sort_values("timestamp")
train_df = train_df.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")

# 3. Build numerical index mappings strictly from the training split
train_users = train_df["user_id"].unique()
train_games = train_df["parent_asin"].unique()

user_to_idx = {u: i for i, u in enumerate(train_users)}
game_to_idx = {g: i for i, g in enumerate(train_games)}

train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
train_df["game_idx"] = train_df["parent_asin"].map(game_to_idx)

# 4. Filter validation and test splits for warm-start evaluation (known entities)
valid_cleaned = valid_df[
    valid_df["user_id"].isin(user_to_idx) & valid_df["parent_asin"].isin(game_to_idx)
].copy()
valid_cleaned["user_idx"] = valid_cleaned["user_id"].map(user_to_idx)
valid_cleaned["game_idx"] = valid_cleaned["parent_asin"].map(game_to_idx)

test_cleaned = test_df[
    test_df["user_id"].isin(user_to_idx) & test_df["parent_asin"].isin(game_to_idx)
].copy()
test_cleaned["user_idx"] = test_cleaned["user_id"].map(user_to_idx)
test_cleaned["game_idx"] = test_cleaned["parent_asin"].map(game_to_idx)

# Save cleaned files
train_df[["user_id", "parent_asin", "rating", "user_idx", "game_idx"]].to_csv(DATA_DIR / "train_cleaned.csv", index=False)
valid_cleaned[["user_id", "parent_asin", "rating", "user_idx", "game_idx"]].to_csv(DATA_DIR / "valid_cleaned.csv", index=False)
test_cleaned[["user_id", "parent_asin", "rating", "user_idx", "game_idx"]].to_csv(DATA_DIR / "test_cleaned.csv", index=False)

print(f"Cleaned Train rows: {len(train_df):,}")
print(f"Cleaned Valid rows: {len(valid_cleaned):,}")
print(f"Cleaned Test rows:  {len(test_cleaned):,}")
print("Saved: train_cleaned.csv, valid_cleaned.csv, test_cleaned.csv")




# PS C:\Users\shrey\minor project\recommendation project> python data_exploration_and_cleaning.py
# --- Step 1: Loading Interaction Datasets ---
# Raw Train Shape: (1045048, 4)
# Raw Valid Shape: (743796, 4)
# Raw Test Shape:  (2766656, 4)

# --- Step 2: Exploratory Data Analysis (EDA) ---
# Unique Users: 329,719
# Unique Games: 83,630
# Matrix Sparsity: 99.996210%

# Rating Distribution in Train:
# rating
# 1.0     75854
# 2.0     48783
# 3.0     89692
# 4.0    171316
# 5.0    659403
# Name: count, dtype: int64
# Imbalance Ratio (Rating >= 4.0): 79.49% of total data
# Mean interactions per user: 3.17
# Median interactions per user: 2
# Users with only 1 rating: 147,504 (44.74%)
# Saved plot to: dataset/eda_rating_distribution.png

# --- Step 3: Cleaning & Index Mapping ---
# Cleaned Train rows: 1,045,048
# Cleaned Valid rows: 318,844
# Cleaned Test rows:  316,273
# Saved: train_cleaned.csv, valid_cleaned.csv, test_cleaned.csv