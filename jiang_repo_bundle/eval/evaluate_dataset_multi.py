from __future__ import annotations

import json, math, csv, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from tqdm import tqdm
from vector_graph_pipeline.code.util.pipeline import run_pipeline
# [ADD] --- helpers for diagnostics ---
import math

def _to_base_id(x: str) -> str:
    # 兼容 '12345#p2' 这类；没有#就原样返回
    if not isinstance(x, str):
        return x
    return x.split('#', 1)[0]

def _first_gold_rank_ids(ids_in_order, gold_base_ids) -> float:
    """在进入重排前的候选序列里，返回首个 gold 的 1-based 名次；若没有，返回 math.inf"""
    gset = set(_to_base_id(g) for g in gold_base_ids)
    for i, cid in enumerate(ids_in_order, start=1):
        base = _to_base_id(cid)
        if base in gset or cid in gset:
            return float(i)
    return math.inf

def _safe_get_id(obj):
    """尽量鲁棒地从候选对象里拿到 chunk_id/base_id/id"""
    # 常见字段：chunk_id / id / base_id / doc_id
    for key in ("chunk_id", "id", "base_id", "doc_id"):
        if isinstance(obj, dict) and key in obj and obj[key]:
            return obj[key]
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val:
                return val
    # 兜底：字符串本身
    if isinstance(obj, str):
        return obj
    return None
# [END ADD]

# -------------------------------
# IO helpers
# -------------------------------
def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _get_field(obj: Dict, keys: List[str], default=None):
    for k in keys:
        if k in obj:
            return obj[k]
    return default

def extract_question_and_id(obj: Dict) -> Tuple[str, str]:
    q = _get_field(obj, ["question", "query", "q", "text", "prompt"], None)
    if not q or not isinstance(q, str):
        raise ValueError("Cannot find 'question' text in item.")
    qid = _get_field(obj, ["id", "qid", "question_id", "uid"], None)
    if qid is None:
        qid = ""
    return q.strip(), str(qid)

# -------------------------------
# Gold extraction (robust, handles string-formatted lists)
# -------------------------------
_NUM_RE = re.compile(r"[A-Za-z0-9]+")

def _parse_list_like_string(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        out: List[str] = []
        if isinstance(obj, list):
            for x in obj:
                if x is None: continue
                if isinstance(x, (int, float)):
                    out.append(str(int(x)))
                elif isinstance(x, str):
                    x = x.strip()
                    if x:
                        out.append(x)
            return out
    except Exception:
        pass
    # Regex fallback
    toks = _NUM_RE.findall(s)
    out = [t for t in toks if not (t.lower() in {"pmid", "id", "doc"})]
    return out

def extract_gold_doc_ids(obj: Dict) -> List[str]:
    gold: List[str] = []

    def push(x):
        if x is None:
            return
        if isinstance(x, str):
            x = x.strip()
            if x:
                gold.append(x)
        elif isinstance(x, (int, float)):
            gold.append(str(int(x)))

    if "relevant_passage_ids" in obj:
        rpi = obj["relevant_passage_ids"]
        if isinstance(rpi, list):
            for it in rpi:
                push(it)
        elif isinstance(rpi, str):
            for it in _parse_list_like_string(rpi):
                push(it)
        if gold:
            return sorted(set(gold))

    # Fallbacks
    for key in ["doc_ids", "docs_gold", "gold_doc_ids", "pmids", "pmid_list"]:
        if key in obj and isinstance(obj[key], list):
            for it in obj[key]:
                push(it)
        if gold:
            return sorted(set(gold))

    for key in ["pmid", "docid", "doc_id"]:
        if key in obj and obj.get(key) is not None:
            push(obj[key])
            return sorted(set(gold))

    return sorted(set(gold))

# -------------------------------
# Normalization
# -------------------------------
def _norm_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'^(pmid:|doc:|id:)', '', s)
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

# -------------------------------
# Build chunk_id -> base_id map from graph
# -------------------------------
def build_chunk2base_from_graph(graph_path: Path) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not graph_path or not graph_path.exists():
        return mp
    with graph_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    edges = data.get("links") or data.get("edges") or []
    for e in edges:
        cid = e.get("chunk_id")
        bid = e.get("base_id")
        if not cid or not bid:
            continue
        mp.setdefault(cid, bid)
    return mp

def pred_doc_ids_from_chunks(pred_chunk_ids: List[str], chunk2base: Dict[str, str], k: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for cid in pred_chunk_ids:
        bid = chunk2base.get(cid)
        bnorm = _norm_id(bid if bid is not None else cid.split("#",1)[0])
        if not bnorm or bnorm in seen:
            continue
        seen.add(bnorm)
        out.append(bnorm)
        if len(out) >= k:
            break
    return out

# -------------------------------
# Metrics
# -------------------------------
def precision_recall_f1_at_k(gold_doc_ids: List[str], pred_chunk_ids: List[str], k: int, chunk2base: Dict[str, str]) -> Tuple[float, float, float, int]:
    if k <= 0:
        return 0.0, 0.0, 0.0, 0
    G = set(_norm_id(x) for x in gold_doc_ids if x)
    if not G:
        return 0.0, 0.0, 0.0, 0
    Pk = pred_doc_ids_from_chunks(pred_chunk_ids, chunk2base, k)
    tp = sum(1 for b in Pk if b in G)
    precision = tp / float(k)
    recall = tp / float(len(G))
    f1 = 0.0 if (precision + recall) == 0.0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, tp

def _agg(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = sum(1 for r in rows if (r.get("tp", 0) or 0) > 0)
    n = len(rows)
    avg_p = sum(r.get("precision", 0.0) for r in rows) / n
    avg_r = sum(r.get("recall", 0.0) for r in rows) / n
    avg_f1 = sum(r.get("f1", 0.0) for r in rows) / n
    return {"hit_rate": hit / n, "precision": avg_p, "recall": avg_r, "f1": avg_f1}

# -------------------------------
# Main evaluation
# -------------------------------
def evaluate_dataset_multi(
    dataset_path: Path,
    out_dir: Path,
    graph_path: Path,
    ks: List[int],
    seed_top_k: int = 5,
    appnp_k_hop: int = 1,
    appnp_top_nodes: int = 300,
    bfs_depth: int = 2,
    bfs_max_chunks: int = 60,
    rerank_top_n: int = 64,
    device: Optional[str] = None,
    embed_model: Optional[Path] = None,
    reranker_model: Optional[str] = None,
    max_samples: Optional[int] = None,
    skip_errors: bool = True,
    filter_gold_in_graph: bool = False,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(int(x) for x in ks if int(x) > 0))
    if not ks:
        raise ValueError("No valid Ks provided.")
    maxK = max(ks)
    print(f"[CFG][IN] seed_top_k={seed_top_k} appnp_top_nodes={appnp_top_nodes} "
          f"bfs_max_chunks={bfs_max_chunks} rerank_top_n={rerank_top_n} maxK={maxK}")

    rerank_top_n = max(rerank_top_n, maxK)

    print(f"[CFG][OUT] seed_top_k={seed_top_k} appnp_top_nodes={appnp_top_nodes} "
          f"bfs_max_chunks={bfs_max_chunks} rerank_top_n={rerank_top_n}")
    rerank_top_n = max(rerank_top_n, maxK)

    items = list(_read_jsonl(dataset_path))
    if max_samples is not None:
        items = items[:int(max_samples)]
    n_total = len(items)
    if n_total == 0:
        raise RuntimeError(f"No items found in {dataset_path}")

    chunk2base = build_chunk2base_from_graph(graph_path)
    print(f"[info] chunk2base mapping loaded: {len(chunk2base)} entries")
    graph_base_ids = set(_norm_id(b) for b in chunk2base.values())

    results_by_k: Dict[int, List[Dict]] = {k: [] for k in ks}
    effective = 0

    pbar = tqdm(range(n_total), desc=f"Evaluating @ks={ks}", ncols=100)
    _diag_total = 0
    _diag_no_gold = 0
    _diag_cutoff_by_budget = 0
    _diag_seen_by_reranker = 0
    for idx in pbar:
        obj = items[idx]
        try:
            qtext, qid = extract_question_and_id(obj)
            if not qid: qid = str(idx)
            gold_docs = extract_gold_doc_ids(obj)

            if filter_gold_in_graph:
                gold_docs = [g for g in gold_docs if _norm_id(g) in graph_base_ids]

            if not gold_docs:
                continue  # skip no-gold sample

            effective += 1

            res = run_pipeline(
                question=qtext,
                seed_top_k=seed_top_k,
                appnp_k_hop=appnp_k_hop,
                appnp_top_nodes=appnp_top_nodes,
                bfs_depth=bfs_depth,
                bfs_max_chunks=bfs_max_chunks,
                rerank_top_n=rerank_top_n,
                device=device,
                embed_model=embed_model,
                reranker_model=reranker_model,
                edge_policy="either",
            )
            # === [ADD-PerQ] diagnostics per question (在调用重排之后、取 preds 之前) ===
            # 1) 取重排前候选的有序列表（按你返回结构的常见键做回退）
            pre_list = (
                res.get("pre_candidates")
                or res.get("pre_ranked")
                or res.get("candidates")
                or res.get("candidate_chunks")
                or []
            )
            pre_ids = []
            for c in pre_list:
                cid = _safe_get_id(c)
                if cid:
                    pre_ids.append(cid)

            # 2) gold 的 base_id 集合（已考虑你前面做过 filter_gold_in_graph）
            G = set(_norm_id(x) for x in gold_docs if x)

            # 3) 计算重排前 gold 的首个名次（使用 chunk2base 做映射，失败再用 '#'-split 兜底）
            pre_rank_gold = math.inf
            for i, cid in enumerate(pre_ids, start=1):
                base = (
                    chunk2base.get(cid)
                    or chunk2base.get(_norm_id(cid))
                    or str(cid).split('#', 1)[0]
                )
                if base is not None:
                    base = _norm_id(base)
                if base in G:
                    pre_rank_gold = float(i)
                    break

            # 4) 是否会被重排“看见”
            seen_by_reranker = (pre_rank_gold <= rerank_top_n)

            # 5) 更新计数与打印（qid 前面你已处理：若无则用 idx）
            _diag_total += 1
            if math.isinf(pre_rank_gold):
                _diag_no_gold += 1
            else:
                if seen_by_reranker:
                    _diag_seen_by_reranker += 1
                else:
                    _diag_cutoff_by_budget += 1

            print(f"[DIAG] Q={qid} pre_rank_gold={'INF' if math.isinf(pre_rank_gold) else int(pre_rank_gold)} | "
                  f"rerank_top_n={rerank_top_n} | seen_by_reranker={seen_by_reranker}")
            # === [END ADD-PerQ] ===
            print(f"[DIAG]   candidates={len(res.get('candidate_chunks') or [])}, "
                  f"reranked={len(res.get('reranked') or [])}, "
                  f"cutoff={(not math.isinf(pre_rank_gold)) and (pre_rank_gold > rerank_top_n)}")
            preds = [x["chunk_id"] for x in (res.get("reranked") or []) if "chunk_id" in x]

            # Warmup diagnostics for first effective sample
            if effective == 1:
                G = set(_norm_id(x) for x in gold_docs if x)
                P6 = pred_doc_ids_from_chunks(preds, chunk2base, 6)
                tp6 = sum(1 for b in P6 if b in G)
                seed_nodes = res.get("num_seed_nodes") or 0
                seed_edges = res.get("num_seed_edges") or 0
                exp_nodes  = res.get("num_expanded_nodes") or 0
                exp_edges  = res.get("num_expanded_edges") or 0
                candidates = len(res.get("candidate_chunks") or [])
                reranked   = len(res.get("reranked") or [])
                print(
                    "\n[Warmup] "
                    f"gold_docs={len(G)}, "
                    f"seed_nodes={seed_nodes}, seed_edges={seed_edges}, "
                    f"expanded_nodes={exp_nodes}, expanded_edges={exp_edges}, "
                    f"candidates={candidates}, reranked={reranked}, "
                    f"hit@6={tp6}/{len(P6)} (doc-level, graph-mapped)\n",
                    flush=True
                )

            # === 每个问题都打印 hit 情况（按各个 k） ===
            Gset = set(_norm_id(x) for x in gold_docs if x)
            per_k_msgs = []
            for k in ks:
                Pk = pred_doc_ids_from_chunks(preds, chunk2base, k)
                tp = sum(1 for b in Pk if b in Gset)
                per_k_msgs.append(f"@{k} {tp}/{len(Pk)}")
            print(f"[Q {idx} | id={qid}] hits: " + "  ".join(per_k_msgs), flush=True)

            for k in ks:
                prec, rec, f1, tp = precision_recall_f1_at_k(gold_docs, preds, k, chunk2base)
                results_by_k[k].append({
                    "idx": idx,
                    "id": qid,
                    "k": k,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tp": tp,
                    "gold_count": len(gold_docs),
                    "pred_count": len(preds),
                })

        except Exception as e:
            if not skip_errors:
                raise
            try:
                gtmp = extract_gold_doc_ids(obj)
                if filter_gold_in_graph:
                    gtmp = [g for g in gtmp if _norm_id(g) in graph_base_ids]
            except Exception:
                gtmp = []
            if gtmp:
                for k in ks:
                    results_by_k[k].append({
                        "idx": idx,
                        "id": str(obj.get("id", idx)),
                        "k": k,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "tp": 0,
                        "gold_count": len(gtmp),
                        "pred_count": 0,
                        "error": str(e),
                    })

    outputs = {}
    for k, rows in results_by_k.items():
        top_n = max(1, math.ceil(0.3 * len(rows)))
        all_sorted = sorted(rows, key=lambda r: r["f1"], reverse=True)
        top_sorted = all_sorted[:top_n]

        def _agg(rows):
            if not rows:
                return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        agg_all = _agg(rows)
        agg_top = _agg(top_sorted)

        out_dir.mkdir(parents=True, exist_ok=True)
        all_csv = out_dir / f"eval_k{k}_all.csv"
        with all_csv.open("w", encoding="utf-8", newline="") as fh:
            import csv as _csv
            w = _csv.writer(fh)
            w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}", "tp", "gold_count", "pred_count"])
            for rank, r in enumerate(all_sorted, start=1):
                w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["tp"], r["gold_count"], r["pred_count"]])

        top_csv = out_dir / f"eval_k{k}_top30.csv"
        with top_csv.open("w", encoding="utf-8", newline="") as fh:
            import csv as _csv
            w = _csv.writer(fh)
            w.writerow(["rank", "idx", "id", f"precision@{k}", f"recall@{k}", f"f1@{k}"])
            for rank, r in enumerate(top_sorted, start=1):
                w.writerow([rank, r["idx"], r["id"], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}"])

        (out_dir / f"eval_k{k}_all.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"eval_k{k}_top30.json").write_text(json.dumps(top_sorted, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = {
            "k": k,
            "num_questions_scored": len(rows),
            "num_questions_effective": len(rows),
            "aggregate_all": {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "aggregate_top30": {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "outputs": {
                "all_csv": str(all_csv),
                "top30_csv": str(top_csv),
                "all_json": str(out_dir / f"eval_k{k}_all.json"),
                "top30_json": str(out_dir / f"eval_k{k}_top30.json"),
            }
        }
        (out_dir / f"eval_k{k}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs[k] = summary

    print("[DIAG][SUMMARY] total_questions={} | no_gold_in_candidates={} | cutoff_by_budget={} | seen_by_reranker={}"
          .format(_diag_total, _diag_no_gold, _diag_cutoff_by_budget, _diag_seen_by_reranker))
    print(f"[DIAG]   candidates={len(res.get('candidate_chunks') or [])}, "
          f"reranked={len(res.get('reranked') or [])}, "
          f"cutoff={(not math.isinf(pre_rank_gold)) and (pre_rank_gold > rerank_top_n)}")

    print(f"[Done] 有效样本(有gold) = {effective} / 总样本 = {n_total}", flush=True)
    return {"ks": ks, "effective": effective, "total": n_total, "summaries": outputs}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="One-pass multi-K (doc-level), graph-mapped; per-question hit printing.")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--ks", type=str, nargs="+", default=[
        "15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"
    ])
    ap.add_argument("--seed-top-k", type=int, default=30)
    ap.add_argument("--appnp-k-hop", type=int, default=1)
    ap.add_argument("--appnp-top-nodes", type=int, default=600)
    ap.add_argument("--bfs-depth", type=int, default=1)
    ap.add_argument("--bfs-max-chunks", type=int, default=192)
    ap.add_argument("--rerank-top-n", type=int, default=96)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embed-model", type=Path, default=None)
    ap.add_argument("--reranker-model", type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--skip-errors", action="store_true", default=True)
    ap.add_argument("--filter-gold-in-graph", action="store_true", default=False)
    args = ap.parse_args()

    print(f"[CLI] seed_top_k={args.seed_top_k} appnp_top_nodes={args.appnp_top_nodes} "
          f"bfs_max_chunks={args.bfs_max_chunks} rerank_top_n={args.rerank_top_n} ks={args.ks}")

    ks: List[int] = []
    for token in args.ks:
        if "," in token:
            ks.extend(int(t) for t in token.split(",") if t.strip())
        else:
            ks.append(int(token))
    ks = sorted(set(ks))

    summary = evaluate_dataset_multi(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        graph_path=args.graph,
        ks=ks,
        seed_top_k=args.seed_top_k,
        appnp_k_hop=args.appnp_k_hop,
        appnp_top_nodes=args.appnp_top_nodes,
        bfs_depth=args.bfs_depth,
        bfs_max_chunks=args.bfs_max_chunks,
        rerank_top_n=args.rerank_top_n,
        device=args.device,
        embed_model=args.embed_model,
        reranker_model=args.reranker_model,
        max_samples=args.max_samples,
        skip_errors=args.skip_errors,
        filter_gold_in_graph=args.filter_gold_in_graph,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
