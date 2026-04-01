import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent))

from parse_query_paper import parse_query_paper
from bm25_retrieval import BM25Retrieval
from Dense import DenseRetrieval
from build_llm_query import generate_llm_query

# ===================== 配置 =====================
CORPUS_PATH = BASE_DIR / "data/processed/papers_mvp.jsonl"
TOP_PER_ROUTE = 20
FINAL_TOP = 10

# ===================== 清理标题，用于文件夹名称 =====================
def clean_folder_name(title):
    title = title.replace("\n", "").replace("\r", "")
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.replace(" ", "_")
    return title[:50]

# ===================== 加载语料 =====================
def load_corpus():
    papers = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    return papers

# ===================== 四条检索路线 =====================
def run_all_routes(corpus, q_ori, q_llm, kw_llm):
    bm25 = BM25Retrieval(corpus)
    dense = DenseRetrieval(corpus, index_dir=BASE_DIR / "index")

    return {
        "bm25_ori": bm25.search(q_ori, TOP_PER_ROUTE),
        "dense_ori": dense.search(q_ori, TOP_PER_ROUTE),
        "bm25_llm_kw": bm25.search(kw_llm, TOP_PER_ROUTE),
        "dense_llm": dense.search(q_llm, TOP_PER_ROUTE),
    }

# ===================== 融合排序 =====================
def fuse_ranks(routes):
    paper_info = defaultdict(lambda: {"score": 0.0, "count": 0, "sources": []})
    weight = {
        "bm25_ori": 1,
        "dense_ori": 1.5,
        "bm25_llm_kw": 1.2,
        "dense_llm": 2.0
    }

    for name, res in routes.items():
        for rank, (p, s) in enumerate(res):
            pid = p["paper_id"]
            paper_info[pid]["score"] += weight[name] / (rank + 1)
            paper_info[pid]["count"] += 1
            paper_info[pid]["sources"].append(name)

    ranked = sorted(paper_info.items(), key=lambda x: (-x[1]["score"], -x[1]["count"]))
    return ranked[:FINAL_TOP]

# ===================== 原因说明 =====================
def build_reason(paper, sources, keywords):
    kw_str = "、".join(keywords[:3]) if keywords else "相关主题"
    source_str = " + ".join(sources)

    return f"来自检索路线：[{source_str}]，匹配关键词：{kw_str}"

# ===================== 给网页调用的接口 =====================
# ===================== 给网页调用的接口 =====================
def run_full_pipeline(paper_path):
    print("===== 开始处理 =====")
    query_data, q_original = parse_query_paper(paper_path)
    q_llm, llm_profile = generate_llm_query(query_data, q_original)
    kw_llm = llm_profile.get("keywords", [])
    kw_str = " ".join(kw_llm)

    corpus = load_corpus()
    routes = run_all_routes(corpus, q_original, q_llm, kw_str)
    top10 = fuse_ranks(routes)
    paper_map = {p["paper_id"]: p for p in corpus}

    # ===================== 自动创建保存目录 =====================
    folder_name = clean_folder_name(query_data["title"])
    run_dir = BASE_DIR / "results" / "runs" / folder_name
    os.makedirs(run_dir, exist_ok=True)

    # ===================== 【关键：app.py 调用也会保存】 =====================
    # 保存 query 对比
    compare_path = run_dir / "query_vs_llm_query.json"
    compare_data = {
        "paper_id": query_data["paper_id"],
        "title": query_data["title"],
        "original_query": q_original,
        "llm_profile": llm_profile,
        "llm_enhanced_query": q_llm
    }
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_data, f, ensure_ascii=False, indent=2)

    # 构造最终结果
    final_output = {
        "query_paper_id": query_data["paper_id"],
        "query_title": query_data["title"],
        "llm_keywords": kw_llm,
        "top10_papers": []
    }

    for i, (pid, info) in enumerate(top10):
        p = paper_map[pid]
        reason = build_reason(p, info["sources"], kw_llm)
        score = round(info["score"], 3)
        final_output["top10_papers"].append({
            "rank": i+1,
            "paper_id": pid,
            "title": p["title"],
            "fusion_score": score,
            "reason": reason
        })

    # ===================== 保存 TOP10 结果 =====================
    final_path = run_dir / "top10_result.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"✅ 结果已保存到：{run_dir}")
    return final_output

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("===== 论文检索：输入一篇 → 输出 TOP10 相关论文 =====")
    paper_path = input("\n请拖入论文文件：").strip().replace('"', '').replace("'", "")

    # 1. 解析 query 论文
    query_data, q_original = parse_query_paper(paper_path)
    paper_id = query_data["paper_id"]
    paper_title = query_data["title"]

    # 2. LLM 增强查询
    q_llm, llm_profile = generate_llm_query(query_data, q_original)
    kw_llm = llm_profile.get("keywords", [])
    kw_str = " ".join(kw_llm)

    # 3. 检索 + 融合
    corpus = load_corpus()
    routes = run_all_routes(corpus, q_original, q_llm, kw_str)
    top10 = fuse_ranks(routes)
    paper_map = {p["paper_id"]: p for p in corpus}

    # ===================== 按【论文标题】创建文件夹 =====================
    folder_name = clean_folder_name(paper_title)
    run_dir = BASE_DIR / "results" / "runs" / folder_name
    os.makedirs(run_dir, exist_ok=True)

    # 保存对比文件
    compare_path = run_dir / "query_vs_llm_query.json"
    compare_data = {
        "paper_id": paper_id,
        "title": paper_title,
        "original_query": q_original,
        "llm_profile": llm_profile,
        "llm_enhanced_query": q_llm
    }
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_data, f, ensure_ascii=False, indent=2)

    # 构造最终输出
    final_output = {
        "query_paper_id": paper_id,
        "query_title": paper_title,
        "llm_keywords": kw_llm,
        "top10_papers": []
    }

    print("\n==================== TOP-10 相关论文 ====================")
    for i, (pid, info) in enumerate(top10):
        p = paper_map[pid]
        reason = build_reason(p, info["sources"], kw_llm)
        score = round(info["score"], 3)

        print(f"\nTop {i+1}")
        print(f"标题：{p['title']}")
        print(f"得分：{score}")
        print(f"原因：{reason}")

        final_output["top10_papers"].append({
            "rank": i + 1,
            "paper_id": pid,
            "title": p["title"],
            "fusion_score": score,
            "reason": reason
        })

    # 保存最终结果
    final_path = run_dir / "top10_result.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n✅ 所有结果已保存至：")
    print(run_dir)