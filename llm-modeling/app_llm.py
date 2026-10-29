import json
import os
from pathlib import Path
import pickle
from dotenv import load_dotenv
import faiss
from google import genai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_DIR = ROOT_DIR / "dataset"
load_dotenv(ROOT_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="LLM Semantic Recommender", layout="wide", page_icon="🤖")
st.title("🤖 LLM Semantic Recommender (FAISS Vector + Gemini Reasoning)")
st.markdown("Individual interactive module running on the **Best Saved Semantic Index**.")

@st.cache_resource
def load_llm_assets():
    best_index = faiss.read_index(str(CURRENT_DIR / "best_faiss_index.bin"))
    with open(CURRENT_DIR / "llm_artifacts.pkl", "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    train_df = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    client = genai.Client(api_key=API_KEY) if API_KEY else None
    return best_index, meta, embedder, train_df, client

best_index, meta, embedder, train_df, client = load_llm_assets()

st.success(f"**Loaded Winning Representation:** `{meta.get('best_model_name', 'Semantic Vector')}`")

tab1, tab2, tab3 = st.tabs(["❄️ Zero-Shot Cold Start (New Users)", "👤 Warm-Start Profile Recommendation", "📊 LLM Ablation Studies"])

with tab1:
    st.subheader("Solve Cold-Start (88% Unseen Test Users)")
    query = st.text_input("Enter your gaming interest, theme, or gameplay description:", "Story-driven space exploration game with base building")
    top_k_cold = st.slider("Top K Recommendations:", 3, 10, 5, key="cold_k")

    if st.button("Generate Cold-Start Recommendations", key="btn_cold"):
        with st.spinner("Searching FAISS index and querying Gemini..."):
            q_vec = embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q_vec)
            _, indices = best_index.search(q_vec, top_k_cold + 5)

            candidates = [{"id": meta["idx_to_game"][i], "title": meta["id_to_title"].get(meta["idx_to_game"][i], "Game")} for i in indices[0]]

            if client:
                cand_str = "\n".join([f"- {c['title']}" for c in candidates])
                prompt = f"Recommend {top_k_cold} games for someone wanting '{query}' from:\n{cand_str}\nReturn strictly a JSON array with 'title' and 'reasoning' keys."
                try:
                    res = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
                    clean = res.text.strip().replace("```json", "").replace("```", "").strip()
                    recs = json.loads(clean)
                except Exception:
                    recs = [{"title": c["title"], "reasoning": "High semantic embedding similarity match."} for c in candidates[:top_k_cold]]
            else:
                recs = [{"title": c["title"], "reasoning": "Offline vector similarity match."} for c in candidates[:top_k_cold]]

            for r in recs:
                st.markdown(f"#### 🎮 {r.get('title')}")
                st.write(f"**Explanation:** {r.get('reasoning')}")
                st.divider()

with tab2:
    active_users = train_df["user_id"].value_counts().head(30).index.tolist()
    user = st.selectbox("Select Active User ID from Dataset:", active_users)
    top_k_warm = st.slider("Top K Recommendations:", 3, 10, 5, key="warm_k")

    if st.button("Generate Profile Recommendations", key="btn_warm"):
        past = train_df[(train_df["user_id"] == user) & (train_df["rating"] >= 4.0)]["parent_asin"].head(5).tolist()
        liked_docs = [meta["id_to_doc"][g] for g in past if g in meta["id_to_doc"]]

        if liked_docs:
            vecs = embedder.encode(liked_docs, convert_to_numpy=True)
            u_vec = np.mean(vecs, axis=0, keepdims=True)
            faiss.normalize_L2(u_vec)
            scores, indices = best_index.search(u_vec, top_k_warm + len(past))

            st.subheader(f"Recommendations for `{user}`:")
            count = 0
            for idx in indices[0]:
                gid = meta["idx_to_game"][idx]
                if gid not in past:
                    title = meta["id_to_title"].get(gid, gid)
                    st.markdown(f"**{count+1}. 🎮 {title}**")
                    st.caption(f"Product ASIN: `{gid}` | Semantic Centroid Similarity")
                    count += 1
                if count == top_k_warm:
                    break

with tab3:
    st.subheader("LLM Semantic Representation Experiments Comparison")
    if (CURRENT_DIR / "llm_experiments_ablation.csv").exists():
        ablation_df = pd.read_csv(CURRENT_DIR / "llm_experiments_ablation.csv")
        st.dataframe(ablation_df, use_container_width=True)
        st.bar_chart(ablation_df.set_index("Experiment")[["Precision@10", "Recall@10", "NDCG@10"]])
    else:
        st.info("Run 03_LLM_Recommender.ipynb to generate ablation metrics.")