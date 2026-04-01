import json
import os
from pathlib import Path
import dashscope
from dashscope import Generation

# ===================== 配置 =====================
BASE_DIR = Path(__file__).parent.parent

PROMPT_PATH = BASE_DIR / "notes" / "llm_prompt_v1.md"

COMPARE_JSON = BASE_DIR / "results/query_vs_llm_query.json"

# 通义千问 KEY
DASHSCOPE_API_KEY = "sk-447b90f888bc4fb89c0783cbf39ed302"
dashscope.api_key = DASHSCOPE_API_KEY

# ===================== 加载Prompt =====================
def load_prompt_template():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ===================== 构造 LLM 输入 =====================
def build_llm_paper_content(query_data):

    title = query_data["title"]
    full_text = query_data.get("full_text", "") 
    abstract = query_data.get("abstract", "")
    intro = query_data.get("intro_text", "")
    conclusion = query_data.get("conclusion_text", "")

    if full_text:
        all_content = full_text
    else:
        all_content = f"摘要：{abstract}\n\n引言：{intro}\n\n结论：{conclusion}"

    return f"""论文标题：{title}

论文全文：
{all_content}
"""

# ===================== 调用通义千问 =====================
def call_tongyi(prompt):
    resp = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        result_format="message"
    )
    if resp.status_code != 200:
        raise Exception(f"API错误：{resp.code} {resp.message}")
    return resp.output.choices[0].message.content.strip()

# ===================== 生成增强 query =====================
def build_llm_retrieval_query(llm_profile):
    desc = llm_profile.get("retrieval_description", "")
    keywords = " ".join(llm_profile.get("keywords", []))
    return f"{desc} {keywords}".strip()

# ===================== 保存对比文件 =====================
def save_query_comparison(query_data, original_query, llm_profile):
    llm_query = build_llm_retrieval_query(llm_profile)
    result = {
        "paper_id": query_data["paper_id"],
        "title": query_data["title"],
        "original_query": original_query,
        "llm_profile": llm_profile,
        "llm_enhanced_query": llm_query
    }
    os.makedirs(os.path.dirname(COMPARE_JSON), exist_ok=True)
    with open(COMPARE_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 对比结果已保存：{COMPARE_JSON}")
    return llm_query

# ===================== 主函数 =====================
def generate_llm_query(query_data, original_query):
    prompt = load_prompt_template()
    content = build_llm_paper_content(query_data)
    full_prompt = prompt.format(paper_content=content)

    print("\n🔍 正在调用通义千问...")
    llm_out = call_tongyi(full_prompt)
    llm_profile = json.loads(llm_out)

    print("\n📊 结构化检索画像：")
    print(json.dumps(llm_profile, indent=2, ensure_ascii=False))

    llm_query = save_query_comparison(query_data, original_query, llm_profile)
    return llm_query, llm_profile