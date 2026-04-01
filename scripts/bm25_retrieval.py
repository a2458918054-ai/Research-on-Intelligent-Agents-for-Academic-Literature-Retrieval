import json
import os
from pathlib import Path
from rank_bm25 import BM25Okapi
import time

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
JSONL_PATH = BASE_DIR / "data" / "processed" / "papers_mvp.jsonl"
RESULT_DIR = BASE_DIR / "results"
TOP_K = 10

# ===================== 分词=====================
def simple_tokenize(text):
    text = text.lower()
    for c in '.,!?;:"()[]{}<>|\\/*-+_=@#$%^&~`':
        text = text.replace(c, " ")
    return [w.strip() for w in text.split() if w.strip()]

# ===================== 加载论文 =====================
def load_papers(jsonl_path):
    papers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    print(f"✅ 加载论文：{len(papers)} 篇")
    return papers

# ===================== 构建 BM25 =====================
def build_bm25(papers):
    corpus = [p["doc_text"] for p in papers]
    tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ BM25 索引构建完成")
    return bm25

# ===================== 检索 =====================
def search_bm25(bm25, papers, query):
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(papers, scores), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K]

# ===================== 保存结果=====================
def save_results(query, results):
    BM25_RESULT_DIR = os.path.join(RESULT_DIR, "BM25")
    os.makedirs(BM25_RESULT_DIR, exist_ok=True)
    
    query_id = f"query_{int(time.time())}"
    result_path = os.path.join(BM25_RESULT_DIR, f"result_{query_id}.json")

    output = []
    for rank, (paper, score) in enumerate(results, 1):
        output.append({
            "query_id": query_id,
            "rank": rank,
            "candidate_id": paper["paper_id"],
            "score": round(float(score), 4),
            "candidate_title": paper["title"]  
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存到 BM25 文件夹：{result_path}")
    return output, result_path

# ===================== 打印结果=====================
def print_results(output):
    print("\n" + "="*100)
    print(f"🔍 TOP-{len(output)} 检索结果")
    print("="*100)
    for item in output:
        print(f"\nTop{item['rank']:2d}")
        print(f"ID:    {item['candidate_id']}")
        print(f"title:  {item['candidate_title']}")  
        print(f"BM25得分:  {item['score']}")

# ===================== 查询模式 =====================
def get_query(papers):
    print("\n请选择查询模式：")
    print("1 手动输入标题+摘要")
    print("2 按论文 ID 查询")
    mode = input("请输入 1 或 2：").strip()

    if mode == "1":
        print("\n==== 手动输入 ====")
        title = input("标题：").strip()
        abstract = input("摘要：").strip()
        return title + " " + abstract

    elif mode == "2":
        pid = input("\n请输入 paper_id：").strip()
        for p in papers:
            if p["paper_id"] == pid:
                print(f"✅ 找到论文")
                return p["title"] + " " + p["abstract"]
        print("❌ 未找到该论文")
    return None
class BM25Retrieval:
    def __init__(self, papers):
        self.papers = papers
        self.bm25 = build_bm25(papers)

    def search(self, query, top_k=10):
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.papers, scores), key=lambda x: x[1], reverse=True)
        return [(p, float(s)) for p, s in ranked[:top_k]]

# ===================== 主程序 =====================
if __name__ == "__main__":
    papers = load_papers(JSONL_PATH)
    bm25 = build_bm25(papers)
    query = get_query(papers)

    if query:
        print(f"\n📝 查询：{query}")
        results = search_bm25(bm25, papers, query)
        output, path = save_results(query, results)
        print_results(output)