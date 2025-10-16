# -*- coding: utf-8 -*-
"""
run_eval_all.py (updated)
- 传递 --graph 给 eval.evaluate_dataset_multi
- 一次性评估 ks=[2,3,4,5,6]
- 可配置数据/输出/图/设备/重排器路径，以及是否过滤不在图中的 gold

使用：
  python D:\score\jiang_repo_bundle_with_eval\jiang_repo_bundle\eval\run_eval_all.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def main():
    # === 可按需修改（缺省值按你当前环境） ===
    DATASET = Path(r"D:\datanew\question-answer-passages_test.filtered.strict——yuanshi.jsonl")
    OUT_DIR = Path(r"D:\kg_out2\eval")
    GRAPH   = Path(r"D:\kg_out2\global_graph.json")

    # 评测与流水线参数
    KS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]                 # 一次性评估这几个 @k
    RERANK_TOP_N = 96                 # 至少应 >= max(KS)
    BFS_MAX_CHUNKS = 192
    DEVICE = "cuda"                  # 或者 "cpu"
    RERANKER_MODEL = None            # 例如 r"D:/models/bge-reranker-large"；None 表示不传（会用 evaluate_dataset_multi 的默认行为）
    MAX_SAMPLES = None               # 比如 50 做小样本试跑；None 则全量
    FILTER_GOLD_IN_GRAPH = True      # 过滤图外 gold（更公平地评估当前图覆盖范围内的效果）

    # === 构造命令 ===
    repo_root = Path(__file__).resolve().parents[1]  # 指向 jiang_repo_bundle 根目录
    ks_token = ",".join(str(k) for k in KS)

    cmd = [
        sys.executable, "-m", "eval.evaluate_dataset_multi",
        "--dataset", str(DATASET),
        "--out-dir", str(OUT_DIR),
        "--graph", str(GRAPH),
        "--ks", ks_token,
        "--rerank-top-n", str(RERANK_TOP_N),
        "--bfs-max-chunks", str(BFS_MAX_CHUNKS),
        "--device", DEVICE,
    ]
    if FILTER_GOLD_IN_GRAPH:
        cmd.append("--filter-gold-in-graph")
    if MAX_SAMPLES is not None:
        cmd += ["--max-samples", str(MAX_SAMPLES)]
    if RERANKER_MODEL:
        cmd += ["--reranker-model", str(RERANKER_MODEL)]

    print("\n=== Running evaluation ===")
    print("CWD:", repo_root)
    print("CMD:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True, cwd=str(repo_root))

if __name__ == "__main__":
    main()
