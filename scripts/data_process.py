import json
import os
from pathlib import Path
from glob import glob

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).parent.parent
RAW_FOLDER = BASE_DIR / "data" / "raw"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "papers_mvp.jsonl"

MAX_INTRO_PARAS = 3
MAX_CONCLUSION_PARAS = 2

# ===================== 提取章节 =====================
def extract_intro_conclusion(sections):
    intro_paras = []
    conclusion_paras = []

    # sections 是字典！
    if not isinstance(sections, dict):
        return intro_paras, conclusion_paras

    for title, content in sections.items():
        lower_title = title.lower()

        # 提取 Introduction
        if "introduction" in lower_title:
            if isinstance(content, dict) and "paragraphs" in content:
                paras = content["paragraphs"]
                if isinstance(paras, list):
                    intro_paras = [p.strip() for p in paras if isinstance(p, str) and p.strip()]

        # 提取 Conclusion
        if "conclusion" in lower_title:
            if isinstance(content, dict) and "paragraphs" in content:
                paras = content["paragraphs"]
                if isinstance(paras, list):
                    conclusion_paras = [p.strip() for p in paras if isinstance(p, str) and p.strip()]

    return intro_paras, conclusion_paras

# ===================== 处理单篇论文 =====================
def process_one_paper(paper, file_id):
    if not isinstance(paper, dict):
        return {
            "paper_id": file_id,
            "title": "",
            "abstract": "",
            "intro_text": "",
            "conclusion_text": "",
            "doc_text": ""
        }

    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    sections = paper.get("sections", {})

    # 提取正文
    intro_paras, conclusion_paras = extract_intro_conclusion(sections)
    intro_text = "\n".join(intro_paras[:MAX_INTRO_PARAS]).strip()
    conclusion_text = "\n".join(conclusion_paras[:MAX_CONCLUSION_PARAS]).strip()

    # 拼接检索文本
    parts = []
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    if intro_text:
        parts.append(intro_text)
    if conclusion_text:
        parts.append(conclusion_text)

    doc_text = "\n\n".join(parts).strip()

    return {
        "paper_id": file_id,
        "title": title,
        "abstract": abstract,
        "intro_text": intro_text,
        "conclusion_text": conclusion_text,
        "doc_text": doc_text
    }

# ===================== 批量加载 + 自动提取完整 ID =====================
def load_all_papers_from_raw():
    all_papers = []
    json_files = glob(os.path.join(RAW_FOLDER, "*.json"))
    print(f"📂 找到 {len(json_files)} 个 JSON 文件")

    for fpath in json_files:
        try:
            fname = os.path.basename(fpath)  # 例如 2502.00008.json
            file_id = fname.replace(".json", "")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                all_papers.append((data, file_id))

        except Exception:
            print(f"⚠️ 跳过损坏文件：{fname}")

    return all_papers

# ===================== 主函数 =====================
def main():
    papers_with_id = load_all_papers_from_raw()
    print(f"\n🚀 总共加载论文：{len(papers_with_id)} 篇")

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for paper, file_id in papers_with_id:
            item = process_one_paper(paper, file_id)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 处理完成！输出：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()