import json
import time
from openai import OpenAI  # 假设 DeepSeek/Qwen 接口兼容 OpenAI SDK

# -------------------------------
# 1. 配置 Qwen API 客户端
# -------------------------------
client = OpenAI(
    api_key="YOUR_QWEN_API_KEY",  # 替换为你的 API Key
    base_url="https://api.deepseek.com"  # Qwen 兼容 DeepSeek API
)

# -------------------------------
# 2. 指令模板：法律三段论
# -------------------------------
PROMPT_TEMPLATE = """
请将以下法律文书文本拆解为三部分：
1. facts: 案件事实描述
2. law_articles: 涉及的法条
3. conclusion: 法院裁决结论

请以 JSON 格式返回，字段名严格为 "facts", "law_articles", "conclusion"，不要输出其他文本。

文书文本：
{document}
"""

# -------------------------------
# 3. 调用 Qwen 生成伪标签
# -------------------------------
def generate_pseudo_label(document: str, retry=3) -> dict:
    prompt = PROMPT_TEMPLATE.format(document=document)
    
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model="qwen-7b-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 生成更稳定
                max_tokens=512
            )
            text = resp.choices[0].message.content.strip()
            # 尝试直接解析 JSON
            return json.loads(text)
        except Exception as e:
            print(f"[WARN] 解析失败, 尝试第 {attempt+1} 次: {e}")
            time.sleep(1)
    return None

# -------------------------------
# 4. 读取原始法律文书
# -------------------------------
def load_documents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# -------------------------------
# 5. 生成伪监督数据集
# -------------------------------
def generate_dataset(documents):
    dataset = []
    for doc in documents:
        label = generate_pseudo_label(doc)
        if label is not None:
            dataset.append({
                "document": doc,
                "facts": label.get("facts", ""),
                "law_articles": label.get("law_articles", ""),
                "conclusion": label.get("conclusion", "")
            })
    return dataset

# -------------------------------
# 6. 保存为 JSONL
# -------------------------------
def save_dataset(dataset, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# -------------------------------
# 7. 主流程
# -------------------------------
if __name__ == "__main__":
    docs = load_documents("law_documents.txt")
    pseudo_dataset = generate_dataset(docs)
    print(f"生成伪标签样本数: {len(pseudo_dataset)}")
    save_dataset(pseudo_dataset, "pseudo_law.jsonl")
