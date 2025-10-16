from __future__ import annotations
from pathlib import Path

# ===== Default paths (Windows-style as per your project) =====
KG_OUT = Path(r"D:\kg_out2")
VECTOR_DIR = KG_OUT / "vector_graph"

# Core inputs
GLOBAL_GRAPH_JSON = KG_OUT / "global_graph.json"         # node-link JSON (undirected logical graph)
INDEX_CSV         = VECTOR_DIR / "index.csv"             # authoritative row mapping for nodes/edges
EDGES_VEC_NPY     = VECTOR_DIR / "edges.vec.npy"         # M x D (float32); preferred already L2-normalized
EDGES_NORM_NPY    = VECTOR_DIR / "edges.norm.npy"        # optional: row-wise L2 norms if not normalized
NODES_VEC_NPY     = VECTOR_DIR / "nodes.vec.npy"         # N x D (float32); optional for graph ops
META_JSON         = VECTOR_DIR / "meta.json"             # {"dim":1024,"normalized":true,"dtype":"float32",...}
CHUNKS_INDEX_JSON = KG_OUT / "chunks_index.json"         # chunk_id -> text (for reranker only)

# Cache & outputs
SCORES_DIR              = VECTOR_DIR / "scores"          # one-shot query similarities cache
OUTPUT_DIR              = KG_OUT
SEED_SUBGRAPH_JSON      = OUTPUT_DIR / "seed_subgraph.json"
APPNP_SUBGRAPH_JSON     = OUTPUT_DIR / "appnp_subgraph.json"

# Split score files (avoid semantic overwrites)
ALL_CHUNK_SIMS_NPY      = OUTPUT_DIR / "all_chunk_similarities.npy"   # aligned to EDGES rows (faster, compact)
ALL_CHUNK_SIMS_META     = OUTPUT_DIR / "all_chunk_similarities.meta.json"
SEED_CHUNK_SCORES_JSON  = OUTPUT_DIR / "seed_chunk_scores.json"       # {chunk_id: score} for Top-K seeds
APPNP_NODE_SCORES_JSON  = OUTPUT_DIR / "appnp_node_scores.json"       # {node_id: score}

RESULT_JSON             = OUTPUT_DIR / "vector_pipeline_result.json"

# Models (query embed + reranker)
EMBED_MODEL_DIR   = Path(r"D:\score\weitiaomoxing2(hao!)")            # your fine-tuned embed model
RERANKER_DIR = "BAAI/bge-reranker-large"


# FlagEmbedding reranker

def ensure_dirs() -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
