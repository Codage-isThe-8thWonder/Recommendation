import gzip
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("dataset")
train_cleaned_path = DATA_DIR / "train_cleaned.csv"
meta_gz_path = DATA_DIR / "meta_Video_Games.jsonl.gz"  # Apni gzipped file ka exact naam check kar lena
output_path = DATA_DIR / "llm_meta_cleaned.csv"

print("--- Step 1: Loading Training Games to Filter Metadata ---")
train_df = pd.read_csv(train_cleaned_path)
active_game_ids = set(train_df["parent_asin"].unique())
print(f"Active unique games in training: {len(active_game_ids):,}")

def clean_html_and_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)        # Strip HTML tags
    text = re.sub(r"http\S+|www\S+", "", text) # Strip links
    text = re.sub(r"\s+", " ", text)          # Normalize spaces
    return text.strip()

print("\n--- Step 2: Streaming and Parsing .jsonl.gz Metadata ---")
extracted_data = []

with gzip.open(meta_gz_path, "rt", encoding="utf-8") as f:
    for line in tqdm(f, desc="Streaming .jsonl.gz"):
        try:
            item = json.loads(line.strip())
            parent_asin = item.get("parent_asin", item.get("asin", ""))
            
            # Sirf wahi games rakho jo training set me active hain
            if parent_asin in active_game_ids:
                title = clean_html_and_text(item.get("title", "Unknown Title"))
                categories = clean_html_and_text(" > ".join(item.get("category", [])))
                description = clean_html_and_text(" ".join(item.get("description", [])))
                
                extracted_data.append({
                    "parent_asin": parent_asin,
                    "title": title if title else "Unknown Title",
                    "categories": categories if categories else "Video Games",
                    "description": description
                })
        except json.JSONDecodeError:
            continue

meta_df = pd.DataFrame(extracted_data)
meta_df.drop_duplicates(subset=["parent_asin"], inplace=True)

print(f"\nMatched metadata records: {len(meta_df):,}")

# Construct unified text description for LLM semantic embedding
print("Generating 'llm_doc' semantic context string...")
meta_df["llm_doc"] = (
    "Title: " + meta_df["title"] +
    " | Category: " + meta_df["categories"] +
    " | Details: " + meta_df["description"].str.slice(0, 200)
)

meta_df.to_csv(output_path, index=False)
print(f"Saved final enriched metadata to: {output_path}")




# PS C:\Users\shrey\minor project\recommendation project> python meta_text_preprocessing.py
# --- Step 1: Loading Training Games to Filter Metadata ---
# Active unique games in training: 83,630

# --- Step 2: Streaming and Parsing .jsonl.gz Metadata ---
# Streaming .jsonl.gz: 137269it [00:06, 20379.88it/s]

# Matched metadata records: 83,630
# Generating 'llm_doc' semantic context string...
# Saved final enriched metadata to: dataset\llm_meta_cleaned.csv