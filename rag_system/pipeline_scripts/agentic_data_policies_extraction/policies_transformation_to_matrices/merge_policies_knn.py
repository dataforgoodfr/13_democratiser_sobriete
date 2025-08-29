#!/usr/bin/env python3
"""
Semantic clustering of policy texts with scalable ANN + medoid canonicalization.

- Embeds policies with SentenceTransformers (L2-normalized).
- Builds a FAISS inner-product (cosine) k-NN graph.
- Connects edges above a similarity threshold; clusters = connected components.
- Chooses one canonical text per cluster via **medoid** (max avg cosine similarity).
- Returns the original DataFrame with 'cluster_id' and 'policy_canonical'.

Dependencies:
  pip install pandas numpy sentence-transformers faiss-cpu
"""

import re
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# -------- CONFIG --------
class Config:
    text_col: str = "policy"
    model_name: str = "all-MiniLM-L6-v2"             # or "paraphrase-multilingual-MiniLM-L12-v2"
    use_multilingual: bool = False                   # set True to override model_name
    batch_size: int = 32                             # Reduced batch size to avoid memory issues
    max_neighbors: int = 10                          # Reduced k for k-NN to avoid memory issues
    sim_threshold: float = 0.78                      # merge threshold (cosine). Try 0.75–0.85
    normalize_text_flag: bool = True
    random_seed: int = 42
    medoid_exact_max_cluster: int = 100              # Reduced max cluster size

CFG = Config()
if CFG.use_multilingual:
    CFG.model_name = "paraphrase-multilingual-MiniLM-L12-v2"

np.random.seed(CFG.random_seed)

# -------- Text normalization (lightweight) --------
_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s\-\%\€/£$]")  # keep %, currencies, hyphen
def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = _ws.sub(" ", s)
    s = _punct.sub(" ", s)
    return s

# -------- Embedding --------
_model_cache = {}

def embed_texts(texts: List[str], model_name: str, batch_size: int = 512) -> np.ndarray:
    try:
        if model_name not in _model_cache:
            print(f"Loading model: {model_name}")
            _model_cache[model_name] = SentenceTransformer(model_name)
        
        model = _model_cache[model_name]
        # normalize_embeddings=True -> vectors L2-normalized; dot == cosine
        embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(embs, dtype=np.float32)
    except Exception as e:
        print(f"Error in embedding: {e}")
        print("Falling back to simple TF-IDF embeddings...")
        return embed_texts_tfidf(texts)

def embed_texts_tfidf(texts: List[str]) -> np.ndarray:
    """Fallback to simple TF-IDF embeddings when sentence transformers fail"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense array and normalize
    embs = tfidf_matrix.toarray().astype(np.float32)
    embs = normalize(embs, norm='l2', axis=1)
    
    return embs

# -------- FAISS k-NN --------
def build_knn_index(embs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import faiss  # faiss-cpu
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product = cosine (since vectors normalized)
        index.add(embs)
        sims, idx = index.search(embs, k)
        return idx, sims  # shapes: (N, k)
    except Exception as e:
        print(f"FAISS failed, falling back to scikit-learn: {e}")
        return build_knn_index_sklearn(embs, k)

def build_knn_index_sklearn(embs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback k-NN implementation using scikit-learn"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Since embeddings are already normalized, cosine similarity = dot product
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nbrs.fit(embs)
    
    # Get distances and indices
    distances, indices = nbrs.kneighbors(embs)
    
    # Convert distances to similarities (cosine distance = 1 - cosine similarity)
    similarities = 1 - distances
    
    return indices, similarities

# -------- Union-Find --------
class DSU:
    def __init__(self, n:int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x:int)->int:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self, a:int, b:int):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]:
            self.p[ra]=rb
        elif self.r[ra]>self.r[rb]:
            self.p[rb]=ra
        else:
            self.p[rb]=ra
            self.r[ra]+=1

# -------- Clustering from k-NN graph --------
def cluster_from_knn(idx: np.ndarray, sims: np.ndarray, threshold: float) -> np.ndarray:
    """
    Add undirected edges i--j for neighbors with similarity >= threshold,
    then take connected components.
    """
    n = idx.shape[0]
    dsu = DSU(n)
    for i in range(n):
        neighs = idx[i]
        scores = sims[i]
        # skip self at position 0
        for j, s in zip(neighs[1:], scores[1:]):
            if s >= threshold:
                dsu.union(i, int(j))
    # Relabel components to 0..C-1
    roots = [dsu.find(i) for i in range(n)]
    root_to_label: Dict[int,int] = {}
    labels = np.empty(n, dtype=np.int32)
    next_label = 0
    for i, r in enumerate(roots):
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]
    return labels

# -------- Medoid selection --------
def choose_medoid_indices(indices: np.ndarray, embs: np.ndarray) -> int:
    """
    Exact medoid: index (in the ORIGINAL dataframe space) whose embedding has
    the highest average cosine similarity to others in the same cluster.

    Warning: O(m^2) per cluster. Guarded by CFG.medoid_exact_max_cluster.
    """
    m = len(indices)
    if m == 1:
        return int(indices[0])

    if m <= CFG.medoid_exact_max_cluster:
        sub = embs[indices]                          # (m, d), normalized
        # cosine similarity matrix via dot product
        sim = sub @ sub.T                            # (m, m)
        avg = sim.mean(axis=1)                       # includes self (adds 1/m constant to all)
        pick_local = int(np.argmax(avg))
        return int(indices[pick_local])

    # Fallback for very large clusters: centroid-closest (approx medoid, O(m))
    sub = embs[indices]                              # (m, d)
    centroid = sub.mean(axis=0)
    # Re-normalize centroid to compare with cosine
    norm = np.linalg.norm(centroid) + 1e-12
    centroid = centroid / norm
    sims = sub @ centroid                            # (m,)
    pick_local = int(np.argmax(sims))
    return int(indices[pick_local])

# -------- Main pipeline --------
def merge_policies_semantic_medoid(df: pd.DataFrame,
                                   text_col: str = None,
                                   model_name: str = None,
                                   max_neighbors: int = None,
                                   sim_threshold: float = None,
                                   batch_size: int = None,
                                   normalize_text_flag: bool = None) -> pd.DataFrame:
    """
    Adds 'cluster_id' and 'policy_canonical' to df and returns it.
    Canonical selection = medoid per cluster.
    """
    try:
        # Params
        text_col = text_col or CFG.text_col
        model_name = model_name or CFG.model_name
        max_neighbors = max_neighbors or CFG.max_neighbors
        sim_threshold = sim_threshold or CFG.sim_threshold
        batch_size = batch_size or CFG.batch_size
        normalize_text_flag = normalize_text_flag if normalize_text_flag is not None else CFG.normalize_text_flag

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        # Work on a copy with a clean RangeIndex to align rows <-> embedding rows
        df = df.copy().reset_index(drop=True)

        # Prepare texts
        if normalize_text_flag:
            df["_policy_norm"] = df[text_col].astype(str).map(normalize_text)
            texts = df["_policy_norm"].tolist()
        else:
            texts = df[text_col].astype(str).fillna("").tolist()

        # Embed
        print(f"Embedding {len(texts)} texts with '{model_name}'...")
        embs = embed_texts(texts, model_name=model_name, batch_size=batch_size)  # (N, d), normalized

        # k-NN graph via FAISS
        k = min(max_neighbors, max(2, len(texts)))
        print(f"Building FAISS k-NN (k={k}) and clustering with threshold={sim_threshold:.2f}...")
        knn_idx, knn_sims = build_knn_index(embs, k=k)

        # Graph clustering
        labels = cluster_from_knn(knn_idx, knn_sims, threshold=sim_threshold)
        df["cluster_id"] = labels

        # Medoid per cluster
        print("Selecting canonical medoid per cluster...")
        canonical_map: Dict[int, str] = {}
        for cid, grp in df.groupby("cluster_id", sort=False):
            idx = grp.index.to_numpy()                   # these are 0..N-1 after reset_index
            medoid_row = choose_medoid_indices(idx, embs)
            canonical_map[int(cid)] = df.loc[medoid_row, text_col]

        df["policy_canonical"] = df["cluster_id"].map(canonical_map)

        # Cleanup
        if "_policy_norm" in df.columns:
            df.drop(columns=["_policy_norm"], inplace=True)

        return df
        
    except Exception as e:
        print(f"Error in merge_policies_semantic_medoid: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up model cache to free memory
        global _model_cache
        try:
            for model in _model_cache.values():
                del model
            _model_cache.clear()
            print("Model cache cleaned up")
        except:
            pass

# -------- Example / CLI --------
if __name__ == "__main__":
    try:
        # Example input; replace with your load path.
        example = pd.DataFrame({
            "policy": [
                "Ban single-use plastics nationwide by 2028.",
                "Prohibit disposable plastic items by 2028 (national law).",
                "Introduce a carbon tax at $40/ton, rising annually.",
                "Implement a $40 per ton carbon levy with yearly increase.",
                "Raise the federal minimum wage to $15/hr.",
                "Increase minimum wage to $15 per hour nationwide.",
                "Oppose tax increases on small businesses.",
                "No new taxes for small businesses.",
                "Subsidize green energy R&D grants.",
                "Provide tax credits for renewable energy projects."
            ]
        })

        print("Starting policy merging process...")
        out = merge_policies_semantic_medoid(example)
        
        # Quick view
        summary = (out.groupby(["cluster_id","policy_canonical"])
                     .size()
                     .reset_index(name="count")
                     .sort_values(["cluster_id","count"]))
        print(out)
        print("\nCluster summary:\n", summary)
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        try:
            for model in _model_cache.values():
                del model
            _model_cache.clear()
            print("Final model cleanup completed")
        except:
            pass
