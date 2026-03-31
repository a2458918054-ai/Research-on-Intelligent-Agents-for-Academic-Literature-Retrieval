import json
import os
import numpy as np
import time
from pathlib import Path

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
JSONL_PATH = BASE_DIR / "data/processed/papers_mvp.jsonl"

DENSE_VEC_PATH = BASE_DIR / "index/dense_vectors.npy"
DENSE_ID_PATH = BASE_DIR / "index/dense_paper_ids.npy"

RESULT_DIR = BASE_DIR / "results/Dense"
TOP_K = 10

# ===================== 加载论文 =====================
def load_papers(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f"✅ 加载论文：{len(papers)} 篇")
    return papers

# ===================== 构建/加载索引 =====================
def build_dense_index(papers):
    if os.path.exists(DENSE_VEC_PATH) and os.path.exists(DENSE_ID_PATH):
        print("📂 从 index 文件夹加载索引...")
        vecs = np.load(DENSE_VEC_PATH)
        paper_ids = np.load(DENSE_ID_PATH, allow_pickle=True)
        return vecs, paper_ids

    print("🧠 首次生成稠密向量索引...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    doc_texts = [p["doc_text"] for p in papers]
    vecs = model.encode(doc_texts, show_progress_bar=True)
    paper_ids = [p["paper_id"] for p in papers]

    os.makedirs(os.path.dirname(DENSE_VEC_PATH), exist_ok=True)
    np.save(DENSE_VEC_PATH, vecs)
    np.save(DENSE_ID_PATH, paper_ids)
    print("💾 索引已保存")
    return vecs, paper_ids

# ===================== Dense 检索核心 =====================
def search_dense(vecs, paper_ids, papers, query):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    q_vec = model.encode([query])[0]
    norm_vecs = np.linalg.norm(vecs, axis=1)
    norm_q = np.linalg.norm(q_vec)
    similarities = np.dot(vecs, q_vec) / (norm_vecs * norm_q + 1e-10)

    top_indices = np.argsort(similarities)[::-1][:TOP_K]
    results = []
    for idx in top_indices:
        pid = paper_ids[idx]
        paper = next(p for p in papers if p["paper_id"] == pid)
        results.append((paper, float(similarities[idx])))
    return results

# ===================== 保存结果 =====================
def save_results(query, results):
    os.makedirs(RESULT_DIR, exist_ok=True)
    query_id = f"query_{int(time.time())}"
    save_path = os.path.join(RESULT_DIR, f"{query_id}.json")

    output = []
    for rank, (paper, score) in enumerate(results, 1):
        output.append({
            "rank": rank,
            "paper_id": paper["paper_id"],
            "score": round(score, 4),
            "title": paper["title"][:80]
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Dense 结果已保存：{save_path}")
    return output

# ===================== 查询模式 =====================
def get_query(papers):
    print("\n==== Dense 单独检索 ====")
    print("1 手动输入查询")
    print("2 按论文 ID 查询")
    mode = input("请选择 1 或 2：").strip()

    if mode == "1":
        title = input("标题：").strip()
        abstract = input("摘要：").strip()
        return title + " " + abstract

    if mode == "2":
        pid = input("输入论文 ID：").strip()
        for p in papers:
            if p["paper_id"] == pid:
                print(f"✅ 找到论文：{p['title']}")
                return p["title"] + " " + p["abstract"]
        print("❌ 未找到该论文")
    return None

# ===================== 主程序 =====================
if __name__ == "__main__":
    papers = load_papers(JSONL_PATH)
    vecs, paper_ids = build_dense_index(papers)

    query = get_query(papers)
    if not query:
        exit()

    print(f"\n📝 查询内容：{query[:100]}...")
    print("\n🔍 正在进行 Dense 语义检索...")

    results = search_dense(vecs, paper_ids, papers, query)
    save_results(query, results)

    print("\n" + "=" * 70)
    print("📄 Dense 检索 Top10 结果")
    print("=" * 70)
    for i, (p, s) in enumerate(results, 1):
        print(f"Top{i:2d} | {p['title'][:60]}")