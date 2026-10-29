import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path("dataset")
train_cleaned_path = DATA_DIR / "train_cleaned.csv"
meta_path = DATA_DIR / "meta.csv"

print("--- Step 1: Loading Metadata & Aligning with Train Set ---")
train_df = pd.read_csv(train_cleaned_path)
active_game_ids = set(train_df["parent_asin"].unique())
print(f"Active unique games in training: {len(active_game_ids):,}")

# Clean regex helper function
def clean_html_and_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)        # Strip HTML tags
    text = re.sub(r"http\S+|www\S+", "", text) # Strip links
    text = re.sub(r"\s+", " ", text)          # Normalize spaces
    return text.strip()

# Load or generate baseline metadata
if meta_path.exists():
    print("Loading existing meta.csv...")
    meta_df = pd.read_csv(meta_path)
    item_col = "parent_asin" if "parent_asin" in meta_df.columns else "asin"
    # Keep only active games in train interactions
    meta_df = meta_df[meta_df[item_col].isin(active_game_ids)].copy()
    meta_df["parent_asin"] = meta_df[item_col]
else:
    print("meta.csv not found locally. Synthesizing catalog frame for active games...")
    meta_df = pd.DataFrame({"parent_asin": list(active_game_ids)})
    meta_df["title"] = meta_df["parent_asin"].apply(lambda x: f"Video Game Item {x}")
    meta_df["categories"] = "Video Games"
    meta_df["description"] = "Interactive entertainment digital game software."

# Ensure required fields exist
for col in ["title", "categories", "description"]:
    if col not in meta_df.columns:
        meta_df[col] = ""

print("\n--- Step 2: Cleaning Metadata Attributes ---")
meta_df["title"] = meta_df["title"].fillna("Unknown Title").apply(clean_html_and_text)
meta_df["categories"] = meta_df["categories"].fillna("Video Games").apply(clean_html_and_text)
meta_df["description"] = meta_df["description"].fillna("").apply(clean_html_and_text)

# Construct unified text description for LLM semantic embedding
print("Generating 'llm_doc' semantic context string...")
meta_df["llm_doc"] = (
    "Title: " + meta_df["title"] +
    " | Category: " + meta_df["categories"] +
    " | Details: " + meta_df["description"].str.slice(0, 200)
)

final_meta = meta_df[["parent_asin", "title", "categories", "description", "llm_doc"]].drop_duplicates("parent_asin")
final_meta.to_csv(DATA_DIR / "llm_meta_cleaned.csv", index=False)

print(f"Cleaned metadata records: {len(final_meta):,}")
print("Saved final enriched metadata to: dataset/llm_meta_cleaned.csv")




# PS C:\Users\shrey\minor project\recommendation project> python meta_text_preprocessing.py 
# --- Step 1: Loading Metadata & Aligning with Train Set ---
# Active unique games in training: 83,630
# meta.csv not found locally. Synthesizing catalog frame for active games...

# --- Step 2: Cleaning Metadata Attributes ---
# Generating 'llm_doc' semantic context string...
# Cleaned metadata records: 83,630
# Saved final enriched metadata to: dataset/llm_meta_cleaned.csv