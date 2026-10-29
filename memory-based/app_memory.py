from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = ROOT_DIR / "dataset"

st.set_page_config(page_title="Memory-Based CF Recommender", layout="wide", page_icon="👥")
st.title("👥 Memory-Based Recommender (Item-Item Collaborative Filtering)")
st.markdown("Individual interactive module running on the **Best Saved Similarity Matrix**.")

@st.cache_resource
def load_memory_assets():
    best_sim = load_npz(CURRENT_DIR / "best_memory_model.npz")
    user_item = load_npz(CURRENT_DIR / "user_item_matrix.npz")
    with open(CURRENT_DIR / "memory_artifacts.pkl", "rb") as f:
        meta = pickle.load(f)

    train_df = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    meta_df = pd.read_csv(DATA_DIR / "llm_meta_cleaned.csv")
    id_to_title = dict(zip(meta_df["parent_asin"], meta_df["title"]))
    return best_sim, user_item, meta, train_df, id_to_title

best_sim, user_item, meta, train_df, id_to_title = load_memory_assets()

st.success(f"**Loaded Winning Algorithm:** `{meta.get('best_model_name', 'Item-Item CF')}`")

tab1, tab2 = st.tabs(["🎮 Recommend for Active User", "📊 Experiments & Ablation Study"])

with tab1:
    active_users = train_df["user_id"].value_counts().head(30).index.tolist()
    c1, c2 = st.columns([3, 1])
    with c1:
        user = st.selectbox("Select Active User ID from Dataset:", active_users)
    with c2:
        top_k = st.slider("Top K Recommendations:", 3, 15, 5)

    if st.button("Generate Recommendations", key="btn_mem"):
        u_idx = meta["user_to_idx"][user]
        u_ratings = user_item[u_idx].toarray().flatten()
        scores = best_sim.dot(u_ratings)

        seen = np.where(u_ratings > 0)[0]
        scores[seen] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        st.subheader(f"Top {top_k} Recommendations for User `{user}`:")
        for rank, idx in enumerate(top_indices, 1):
            gid = meta["idx_to_game"][idx]
            title = id_to_title.get(gid, gid)
            st.markdown(f"**{rank}. 🎮 {title}**")
            st.caption(f"Product ASIN: `{gid}` | Match Score: `{scores[idx]:.4f}`")

with tab2:
    st.subheader("Memory-Based Architectural Experiments Comparison")
    if (CURRENT_DIR / "memory_experiments_ablation.csv").exists():
        ablation_df = pd.read_csv(CURRENT_DIR / "memory_experiments_ablation.csv")
        st.dataframe(ablation_df, use_container_width=True)
        st.bar_chart(ablation_df.set_index("Experiment")[["Precision@10", "Recall@10", "NDCG@10"]])
    else:
        st.info("Run 02_Memory_Based_CF.ipynb to generate ablation metrics.")