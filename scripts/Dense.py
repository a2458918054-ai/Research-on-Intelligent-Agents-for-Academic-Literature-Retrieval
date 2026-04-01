import json
import os
import numpy as np
from pathlib import Path

# ===================== 核心类：DenseRetrieval=====================
class DenseRetrieval:
    def __init__(self, papers, index_dir=None):
        self.papers = papers
        self.BASE_DIR = Path(__file__).parent.parent

        if index_dir is None:
            self.DENSE_VEC_PATH = self.BASE_DIR / "index/dense_vectors.npy"
            self.DENSE_ID_PATH = self.BASE_DIR / "index/dense_paper_ids.npy"
        else:
            self.DENSE_VEC_PATH = Path(index_dir) / "dense_vectors.npy"
            self.DENSE_ID_PATH = Path(index_dir) / "dense_paper_ids.npy"

        self.TOP_K = 10
        self.vecs, self.paper_ids = self.build_dense_index()

        # 🔥 🔥 🔥 修复关键：提前加载模型，避免 meta tensor 错误
        self._load_model()

    # ===================== 加载模型（修复版）=====================
    def _load_model(self):
        import torch
        from sentence_transformers import SentenceTransformer

        # 强制使用 CPU，禁止 meta 设备
        torch.set_num_threads(1)
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True,
            device="cpu"  # 强制 CPU，彻底解决 meta tensor
        )

    # ===================== 构建/加载索引 =====================
    def build_dense_index(self):
        if os.path.exists(self.DENSE_VEC_PATH) and os.path.exists(self.DENSE_ID_PATH):
            print("✅ 从 index 文件夹加载稠密向量索引...")
            vecs = np.load(self.DENSE_VEC_PATH)
            paper_ids = np.load(self.DENSE_ID_PATH, allow_pickle=True)
            return vecs, paper_ids

        print("🧠 首次生成稠密向量索引...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True,
            device="cpu"
        )

        doc_texts = [p["doc_text"] for p in self.papers]
        vecs = model.encode(doc_texts, show_progress_bar=True)
        paper_ids = [p["paper_id"] for p in self.papers]

        os.makedirs(os.path.dirname(self.DENSE_VEC_PATH), exist_ok=True)
        np.save(self.DENSE_VEC_PATH, vecs)
        np.save(self.DENSE_ID_PATH, paper_ids)
        print("💾 稠密向量索引已保存到 index 文件夹")
        return vecs, paper_ids

    # ===================== Dense 检索核心 =====================
    def search(self, query, top_k=None):
        if top_k is None:
            top_k = self.TOP_K

        # 🔥 使用已加载好的模型，不再触发错误
        q_vec = self.model.encode([query])[0]

        norm_vecs = np.linalg.norm(self.vecs, axis=1)
        norm_q = np.linalg.norm(q_vec)
        similarities = np.dot(self.vecs, q_vec) / (norm_vecs * norm_q + 1e-10)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            pid = self.paper_ids[idx]
            paper = next((p for p in self.papers if p["paper_id"] == pid), None)
            if paper is not None:
                results.append((paper, float(similarities[idx])))
        return results

# ===================== 以下是你原来的独立运行逻辑 =====================
def load_papers(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f"✅ 加载论文：{len(papers)} 篇")
    return papers

def save_results(query, results):
    BASE_DIR = Path(__file__).parent.parent
    RESULT_DIR = BASE_DIR / "results/Dense"
    os.makedirs(RESULT_DIR, exist_ok=True)
    query_id = f"query_{int(np.random.randint(100000))}"
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

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    JSONL_PATH = BASE_DIR / "data/processed/papers_mvp.jsonl"
    papers = load_papers(JSONL_PATH)
    dense = DenseRetrieval(papers)
    query = get_query(papers)
    if not query:
        exit()
    print(f"\n📝 查询内容：{query[:100]}...")
    print("\n🔍 正在进行 Dense 语义检索...")
    results = dense.search(query)
    save_results(query, results)
    print("\n" + "=" * 70)
    print("📄 Dense 检索 Top10 结果")
    print("=" * 70)
    for i, (p, s) in enumerate(results, 1):
        print(f"Top{i:2d} | {p['title'][:60]}")