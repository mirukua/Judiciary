

import json
import time
from openai import OpenAI  # DeepSeek / Qwen API

# -------------------------------
# 1. 配置 Qwen 客户端
# -------------------------------
client = OpenAI(
    api_key="YOUR_QWEN_API_KEY",
    base_url="https://api.deepseek.com"
)

# -------------------------------
# 2. Qwen 拆解法律文书为 facts / law / conclusion
# -------------------------------
DECOMPOSE_PROMPT = """
请将下面的法律文书拆解为三部分：
1. facts: 案件事实
2. law_articles: 涉及的法条
3. conclusion: 法院裁决结论

要求严格返回 JSON，字段为 "facts", "law_articles", "conclusion"。
文书：
{document}
"""

def qwen_decompose(document: str, retry=3) -> dict:
    prompt = DECOMPOSE_PROMPT.format(document=document)
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model="qwen-7b-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            print(f"[WARN] JSON解析失败, 第 {attempt+1} 次: {e}")
            time.sleep(1)
    return None

# -------------------------------
# 3. 用 Qwen 生成 instruction 多样化
# -------------------------------
INSTRUCTION_PROMPT = """
请根据以下事实生成一条自然语言指令，用于训练法律大模型。
要求：
- 指令描述明确任务
- 用自然语言表达
- 不超过 50 个字

事实：
{facts}
"""

def qwen_generate_instruction(facts: str, retry=3) -> str:
    prompt = INSTRUCTION_PROMPT.format(facts=facts)
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model="qwen-7b-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50
            )
            instruction = resp.choices[0].message.content.strip()
            return instruction
        except Exception as e:
            print(f"[WARN] instruction生成失败, 第 {attempt+1} 次: {e}")
            time.sleep(1)
    return "请根据下文提取案件事实、法条和结论。"

# -------------------------------
# 4. 构建伪监督任务骨架
# -------------------------------
def generate_task_skeleton(document: str) -> dict:
    decomposed = qwen_decompose(document)
    if not decomposed:
        return None

    instruction = qwen_generate_instruction(decomposed["facts"])
    input_text = decomposed["facts"]  # 也可以加 law_articles + 证据摘要
    output_text = json.dumps({
        "facts": decomposed["facts"],
        "law_articles": decomposed["law_articles"],
        "conclusion": decomposed["conclusion"]
    }, ensure_ascii=False)

    return {
        "document": document,
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

# -------------------------------
# 5. 主流程
# -------------------------------
def main():
    # 每行一个文书
    with open("law_documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    dataset = []
    for doc in documents:
        task = generate_task_skeleton(doc)
        if task:
            dataset.append(task)
        else:
            print("[WARN] 文书解析失败，跳过")

    # 保存 JSONL
    with open("pseudo_law_qwen.jsonl", "w", encoding="utf-8") as f:
        for t in dataset:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"生成 {len(dataset)} 条 Qwen 伪监督样本完成。")

if __name__ == "__main__":
    main()
