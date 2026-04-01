import json
import os
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
import time

# 镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
JSONL_PATH = BASE_DIR / "data/processed/papers_mvp.jsonl"
DENSE_VEC_PATH = BASE_DIR / "index/dense_vectors.npy"
DENSE_ID_PATH = BASE_DIR / "index/dense_paper_ids.npy"
RESULT_DIR = BASE_DIR / "results"
TOP_K = 10

# ===================== 分词 =====================
def simple_tokenize(text):
    text = text.lower()
    for c in '.,!?;:"()[]{}<>|\\/*-+_=@#$^&~`':
        text = text.replace(c, " ")
    return [w.strip() for w in text.split() if w.strip()]

# ===================== 加载论文 =====================
def load_papers(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f" 加载论文：{len(papers)} 篇")
    return papers

# ===================== BM25 =====================
def build_bm25(papers):
    corpus = [p["doc_text"] for p in papers]
    tokenized = [simple_tokenize(d) for d in corpus]
    return BM25Okapi(tokenized)

def search_bm25(bm25_model, papers, query):
    tokens = simple_tokenize(query)
    scores = bm25_model.get_scores(tokens)
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K]

# ===================== 稠密检索 =====================
def get_dense_index(papers):
    if os.path.exists(DENSE_VEC_PATH) and os.path.exists(DENSE_ID_PATH):
        print("加载向量索引...")
        vecs = np.load(DENSE_VEC_PATH)
        ids = np.load(DENSE_ID_PATH, allow_pickle=True)
        return vecs, ids

    print(" 生成向量（第一次运行）...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    docs = [p["doc_text"] for p in papers]
    vecs = model.encode(docs, show_progress_bar=True)
    ids = [p["paper_id"] for p in papers]

    os.makedirs(os.path.dirname(DENSE_VEC_PATH), exist_ok=True)
    np.save(DENSE_VEC_PATH, vecs)
    np.save(DENSE_ID_PATH, ids)
    print(" 向量已保存")
    return vecs, ids

def search_dense(vecs, ids, papers, query):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    q_vec = model.encode([query])[0]

    sims = np.dot(vecs, q_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec))
    top_idx = np.argsort(sims)[::-1][:TOP_K]

    res = []
    for i in top_idx:
        pid = ids[i]
        p = next(x for x in papers if x["paper_id"] == pid)
        res.append((p, float(sims[i])))
    return res

# ===================== 保存结果 =====================
def save_results(query, results, method):
    if method == "bm25":
        save_dir = os.path.join(RESULT_DIR, "BM25")
    else:
        save_dir = os.path.join(RESULT_DIR, "Dense")
    
    os.makedirs(save_dir, exist_ok=True)
    query_id = f"query_{int(time.time())}_{method}"
    result_path = os.path.join(save_dir, f"result_{query_id}.json")

    output = []
    for rank, (paper, score) in enumerate(results, 1):
        output.append({
            "query": query[:100],
            "rank": rank,
            "paper_id": paper["paper_id"],
            "score": round(float(score), 4),
            "title": paper["title"]
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n 已保存：{result_path}")
    return output

# ===================== 查询模式 =====================
def get_query(papers):
    print("\n==== 选择查询方式 ====")
    print("1 手动输入标题+摘要")
    print("2 按论文 ID 查询")
    mode = input("请输入 1 或 2：").strip()

    if mode == "1":
        print("\n--- 手动输入 ---")
        title = input("标题：").strip()
        abstract = input("摘要：").strip()
        return title + " " + abstract

    elif mode == "2":
        print("\n--- 按论文 ID 查询 ---")
        pid = input("论文 ID：").strip()
        for p in papers:
            if p["paper_id"] == pid:
                print(f" 找到论文：{p['title']}")
                return p["title"] + " " + p["abstract"]
        print(" 未找到该 ID")
        return None

# ===================== 对比展示 =====================
def compare_results(bm25_out, dense_out):
    print("\n" + "=" * 135)
    print(f"{'':<5} {'BM25 结果':<62} | {'Dense 稠密检索结果':<62}")
    print("=" * 135)
    for i in range(TOP_K):
        bm_title = bm25_out[i]["title"][:58] if i < len(bm25_out) else ""
        dn_title = dense_out[i]["title"][:58] if i < len(dense_out) else ""
        print(f"Top{i+1:<2} | {bm_title:<58} | {dn_title:<58}")

    bm_ids = {x["paper_id"] for x in bm25_out}
    dn_ids = {x["paper_id"] for x in dense_out}
    common = len(bm_ids & dn_ids)
    print(f"\n 重叠：{common}/{TOP_K}  重叠率 {common/TOP_K*100:.1f}%")

# ===================== 主程序 =====================
if __name__ == "__main__":
    papers = load_papers(JSONL_PATH)
    bm25 = build_bm25(papers)
    vecs, ids = get_dense_index(papers)

    query = get_query(papers)
    if not query:
        exit()

    print(f"\n 查询：{query[:100]}...")

    # 检索
    bm25_res = search_bm25(bm25, papers, query)
    dense_res = search_dense(vecs, ids, papers, query)

    # 保存
    bm_out = save_results(query, bm25_res, "bm25")
    dn_out = save_results(query, dense_res, "dense")

    # 对比
    compare_results(bm_out, dn_out)