"""
Day1 核心脚本：数据与接口冻结
功能：
1. 原始数据体检
2. 论文标准化 paper_norm
3. 构建 doc_view / chunk_view
4. 生成 dev query 子集
5. 输出接口样例
"""

import json
import os
import random
from pathlib import Path

# ====================== 路径配置======================
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
RAW_DATA_PATH = BASE_DIR / "data" / "processed" / "papers_mvp.jsonl"  
OUTPUT_DIR = BASE_DIR / "data" / "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEV_QUERY_OUTPUT = OUTPUT_DIR / "dev_query_paper_ids.json"
STATS_OUTPUT = OUTPUT_DIR / "data_statistics.json"
PAPER_NORM_EXAMPLE = OUTPUT_DIR / "paper_norm_example.json"
DOC_VIEW_EXAMPLE = OUTPUT_DIR / "doc_view_example.json"
CHUNK_VIEW_EXAMPLE = OUTPUT_DIR / "chunk_view_example.json"

# ====================== 1. 加载原始数据 ======================
def load_raw_papers(file_path=RAW_DATA_PATH):
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers

# ====================== 2. 数据体检：统计缺失率 & 结构 ======================
def audit_raw_papers(papers):
    total = len(papers)
    has_title = 0
    has_abs = 0
    has_sections = 0
    has_refs = 0
    section_counts = []
    para_counts = []

    for p in papers:
        if p.get("title"): has_title +=1
        if p.get("abstract"): has_abs +=1
        if p.get("sections"): has_sections +=1
        if p.get("references"): has_refs +=1

        sections = p.get("sections", [])
        section_counts.append(len(sections))
        paras = sum([len(s.get("paragraphs", [])) for s in sections])
        para_counts.append(paras)

    avg_sections = sum(section_counts)/total if total else 0
    avg_paras = sum(para_counts)/total if total else 0

    stats = {
        "total_papers": total,
        "has_title": has_title,
        "has_abstract": has_abs,
        "has_sections": has_sections,
        "has_references": has_refs,
        "avg_sections_per_paper": round(avg_sections,2),
        "avg_paragraphs_per_paper": round(avg_paras,2)
    }

    with open(STATS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("✅ 数据体检完成：data/processed/data_statistics.json")
    return stats

# ====================== 3. 标准化论文：paper_norm（接口冻结）======================
def normalize_paper(raw_paper):
    sections = raw_paper.get("sections", [])
    references = raw_paper.get("references", [])

    return {
        "paper_id": raw_paper.get("paper_id", ""),
        "title": raw_paper.get("title", ""),
        "abstract": raw_paper.get("abstract", ""),
        "full_text": raw_paper.get("full_text", ""),
        "sections_norm": normalize_sections(sections),
        "references_norm": normalize_references(references),
        "meta": {}
    }

def normalize_sections(sections):
    norm = []
    for idx, sec in enumerate(sections):
        norm.append({
            "section_id": f"sec_{idx}",
            "section_title": sec.get("section_title", f"section_{idx}"),
            "paragraphs": sec.get("paragraphs", [])
        })
    return norm

def normalize_references(refs):
    norm = []
    for idx, r in enumerate(refs):
        norm.append({
            "ref_id": f"ref_{idx}",
            "ref_title": r.get("title", ""),
            "raw_text": r.get("raw_text", "")
        })
    return norm

# ====================== 4. 构建 doc_view（接口冻结）======================
def build_doc_view(paper_norm):
    return {
        "paper_id": paper_norm["paper_id"],
        "doc_title": paper_norm["title"],
        "doc_text": paper_norm["abstract"],
        "abstract": paper_norm["abstract"]
    }

# ====================== 5. 构建 chunk_view + section_role（接口冻结）======================
def infer_section_role(section_title: str):
    """章节角色分类（固定 7 类）"""
    st = section_title.lower()
    if "abstract" in st: return "abstract"
    if "intro" in st: return "introduction"
    if "method" in st or "approach" in st: return "method"
    if "experiment" in st or "experimental" in st: return "experiment"
    if "result" in st: return "result"
    if "conclu" in st: return "conclusion"
    return "other"

def build_chunk_views(paper_norm):
    chunks = []
    pid = paper_norm["paper_id"]
    sections = paper_norm["sections_norm"]

    for sec in sections:
        sid = sec["section_id"]
        stitle = sec["section_title"]
        role = infer_section_role(stitle)
        paras = sec["paragraphs"]

        for cid, para in enumerate(paras):
            text = para.strip()
            if not text: continue
            chunks.append({
                "chunk_id": f"{pid}_{sid}_ck{cid}",
                "paper_id": pid,
                "section_id": sid,
                "section_title": stitle,
                "section_role": role,
                "chunk_index": cid,
                "chunk_text": text,
                "char_len": len(text),
                "token_len": len(text)//4
            })
    return chunks

# ====================== 6. 生成 dev query 子集（20~50 篇）======================
# def build_dev_query_subset(papers, k=30):
#     pids = [p["paper_id"] for p in papers if p.get("paper_id")]
#     random.shuffle(pids)
#     selected = pids[:min(k, len(pids))]
#     with open(DEV_QUERY_OUTPUT, "w", encoding="utf-8") as f:
#         json.dump(selected, f, indent=2)
#     print(f"✅ 开发子集生成完成：{len(selected)} 篇 | {DEV_QUERY_OUTPUT}")
#     return selected

# ====================== 7. 输出接口样例（给 Day2~Day5 用）======================
def save_interface_examples(paper_norm, doc_view, chunks):
    with open(PAPER_NORM_EXAMPLE, "w", encoding="utf-8") as f:
        json.dump(paper_norm, f, indent=2, ensure_ascii=False)
    with open(DOC_VIEW_EXAMPLE, "w", encoding="utf-8") as f:
        json.dump(doc_view, f, indent=2, ensure_ascii=False)
    if chunks:
        with open(CHUNK_VIEW_EXAMPLE, "w", encoding="utf-8") as f:
            json.dump(chunks[0], f, indent=2, ensure_ascii=False)
    print("✅ 接口样例已保存到 data/processed/")

# ====================== 主入口：Day1 全部流程 ======================
if __name__ == "__main__":
    print("=== Day1：数据与接口冻结 开始运行 ===")

    # 1. 加载数据
    papers = load_raw_papers()
    print(f"加载论文：{len(papers)} 篇")

    # 2. 数据体检
    audit_raw_papers(papers)

    # 3. 标准化第一篇论文
    paper_norm = normalize_paper(papers[0])

    # 4. 构建视图
    doc_view = build_doc_view(paper_norm)
    chunks = build_chunk_views(paper_norm)

    # 5. 保存接口样例
    save_interface_examples(paper_norm, doc_view, chunks)

    # # 6. 生成开发子集
    # build_dev_query_subset(papers, k=30)

    print("\n=== Day1 全部完成 ===")
    print("已冻结接口：paper_norm / doc_view / chunk_view / section_role")
    print("已生成：数据统计、接口样例、30 篇开发测试集")