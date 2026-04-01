import json
MAX_INTRO_PARAS = 3
MAX_CONCLUSION_PARAS = 2

# ===================== 提取章节=====================
def extract_intro_conclusion(sections):
    intro_paras = []
    conclusion_paras = []

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

# ===================== 处理单篇 Query 论文=====================
def parse_query_paper(file_path):
    """
    输入：单篇论文的 json 路径
    输出：
        query_data = {
            "paper_id": ...,
            "title": ...,
            "abstract": ...,
            "intro_text": ...,
            "conclusion_text": ...,
            "doc_text": ...
        }
        search_query = title + abstract （你要的检索用query）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    # 自动从文件名取 ID
    filename = file_path.split("\\")[-1].split("/")[-1]
    paper_id = filename.replace(".json", "")

    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    sections = paper.get("sections", {})

    intro_paras, conclusion_paras = extract_intro_conclusion(sections)
    intro_text = "\n".join(intro_paras[:MAX_INTRO_PARAS]).strip()
    conclusion_text = "\n".join(conclusion_paras[:MAX_CONCLUSION_PARAS]).strip()

    # 拼接 doc_text
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

    query_data = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "intro_text": intro_text,
        "conclusion_text": conclusion_text,
        "doc_text": doc_text
    }

    # 检索 query：title + abstract
    search_query = f"{title} {abstract}".strip()

    # 打印日志
    print("\n========== ✅ Query 论文解析完成==========")
    print(f"论文ID: {paper_id}")
    print(f"标题: {title}")
    print(f"摘要: {abstract}")
    print(f"摘要长度: {len(abstract)}")
    # print(f"intro长度: {len(intro_text)}")
    # print(f"conclusion长度: {len(conclusion_text)}")
    # print(f"doc_text长度: {len(doc_text)}")
    print("=====================================================================\n")

    return query_data, search_query