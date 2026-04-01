# import json
# import os
# import sys
# import numpy as np
# from pathlib import Path

# BASE_DIR = Path(__file__).parent.parent
# sys.path.append(str(Path(__file__).parent))

# # 你的模块
# from parse_query_paper import parse_query_paper
# from bm25_retrieval import BM25Retrieval
# from Dense import DenseRetrieval

# # LLM 模块
# from build_llm_query import generate_llm_query

# # ===================== 配置 =====================
# CORPUS_PATH = BASE_DIR / "data/processed/papers_mvp.jsonl"
# INDEX_DIR = BASE_DIR / "index"
# RESULT_DIR = BASE_DIR / "results/final"
# TOP_K = 10

# # ===================== 加载语料 =====================
# def load_corpus():
#     papers = []
#     with open(CORPUS_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 papers.append(json.loads(line))
#     return papers

# # ===================== 分数归一化 =====================
# def normalize_scores(scores):
#     if not scores:
#         return []
#     max_s, min_s = max(scores), min(scores)
#     if max_s == min_s:
#         return [1.0] * len(scores)
#     return [(s - min_s) / (max_s - min_s) for s in scores]

# # ===================== 融合排序 =====================
# def fuse_rerank(bm25_res, dense_res):
#     paper_dict = {}

#     pids_bm25 = [p["paper_id"] for p, s in bm25_res]
#     scores_bm25 = [s for p, s in bm25_res]
#     norm_bm25 = normalize_scores(scores_bm25)

#     for pid, s, ns in zip(pids_bm25, scores_bm25, norm_bm25):
#         paper_dict[pid] = {"bm25_norm": ns, "dense_norm": 0}

#     pids_dense = [p["paper_id"] for p, s in dense_res]
#     scores_dense = [s for p, s in dense_res]
#     norm_dense = normalize_scores(scores_dense)

#     for pid, s, ns in zip(pids_dense, scores_dense, norm_dense):
#         if pid in paper_dict:
#             paper_dict[pid]["dense_norm"] = ns
#         else:
#             paper_dict[pid] = {"bm25_norm": 0, "dense_norm": ns}

#     fused = []
#     for pid, d in paper_dict.items():
#         fused_score = 0.4 * d["bm25_norm"] + 0.6 * d["dense_norm"]
#         fused.append((pid, fused_score))

#     fused_sorted = sorted(fused, key=lambda x: x[1], reverse=True)
#     corpus = load_corpus()
#     paper_map = {p["paper_id"]: p for p in corpus}

#     final = []
#     for pid, score in fused_sorted[:TOP_K]:
#         final.append((paper_map[pid], round(score, 4)))
#     return final

# # ===================== 保存最终结果 =====================
# def save_final(query_data, llm_query, final_list):
#     os.makedirs(RESULT_DIR, exist_ok=True)
#     fname = f"final_{query_data['paper_id']}.json"
#     path = os.path.join(RESULT_DIR, fname)

#     output = {
#         "query_id": query_data["paper_id"],
#         "query_title": query_data["title"],
#         "llm_enhanced_query": llm_query,
#         "top10": [
#             {"rank": i+1, "paper_id": p["paper_id"], "title": p["title"], "score": s}
#             for i, (p, s) in enumerate(final_list)
#         ]
#     }

#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(output, f, ensure_ascii=False, indent=2)
#     print(f"✅ 最终结果已保存：{path}")

# # ===================== 主流程 =====================
# if __name__ == "__main__":
#     print("=" * 70)
#     print("    论文检索 MVP：LLM 增强 + BM25 + Dense 融合排序")
#     print("=" * 70)

#     paper_path = input("\n📄 输入论文路径：").strip().replace('"', "")

#     # 1. 解析论文
#     query_data, original_query = parse_query_paper(paper_path)

#     # 2. LLM 增强查询
#     llm_retrieval_query = generate_llm_query(query_data, original_query)

#     # 3. 加载库
#     corpus = load_corpus()

#     # 4. BM25
#     print("\n🚀 运行 BM25 检索...")
#     bm25 = BM25Retrieval(corpus)
#     bm25_res = bm25.search(llm_retrieval_query, TOP_K)

#     # 5. Dense
#     print("🚀 运行 Dense 检索...")
#     dense = DenseRetrieval(corpus, index_dir=INDEX_DIR)
#     dense_res = dense.search(llm_retrieval_query, TOP_K)

#     # 6. 融合
#     print("\n⚡ 融合排序中...")
#     final_top10 = fuse_rerank(bm25_res, dense_res)

#     # 7. 输出
#     print("\n" + "=" * 80)
#     print(" 📊 最终 TOP10 结果")
#     print("=" * 80)
#     for i, (p, s) in enumerate(final_top10):
#         print(f"Top{i+1:2d} | {p['title'][:70]}")

#     save_final(query_data, llm_retrieval_query, final_top10)
#     print("\n🎉 全部流程完成！")