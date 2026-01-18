import json
import time
import os
from openai import OpenAI  # DeepSeek API 使用 openai SDK 接口风格

# -------------------------------
# 配置 DeepSeek API 客户端
# -------------------------------
client = OpenAI(
    api_key='sk-8ac853ad2fc34453824ea0474b35cdfb',  # 需在环境变量里配置
    base_url="https://api.deepseek.com"
)

# -------------------------------
# 使用 DeepSeek API 重构案件
# -------------------------------
def reconstruct_case_deepseek(case_json_str: str, max_retries: int = 3) -> str:
    """
    使用 DeepSeek API 将原始案件 JSON 字符串重构为法律三段论格式
    支持失败重试
    """
    prompt = f"""
你是一个法律分析助手。
请根据以下 JSON 格式的案件数据进行分析：
- 自动理解字段含义（例如 CaseRecord, JudgeReason, JudgeResult, LegalBasis 等）
- 给出三段式分析：
  1. 大前提：引用适用的法律条文或原则(只需列出条例,无需内容)
  2. 小前提：结合案件事实进行分析
  3. 结论：给出裁判结果或答案
请直接用自然语言概括输出，不需要 JSON 格式。

案件数据：
{case_json_str}
"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的法律分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
                stream=False
            )
            text = response.choices[0].message.content
            return text.strip()
        except Exception as e:
            print(f"API 请求失败，第 {attempt+1} 次重试: {e}")
            time.sleep(1 + attempt*2)  # 指数退避重试

    print("多次请求失败，返回空字符串")
    return ""

# -------------------------------
# 分批处理 JSON
# -------------------------------
def process_json_file_deepseek(input_file: str, output_file: str, batch_size: int = 5):
    """
    读取原始 JSON 文件，对每条案件数据使用 DeepSeek API 重构
    支持分批处理
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    ctxs = data.get("ctxs", {})

    case_items = list(ctxs.items())
    total = len(case_items)
    print(f"总案件数: {total}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if end == 10:
            break
        batch = case_items[start:end]
        print(f"处理第 {start+1}-{end} 条案件...")

        for key, case in batch:
            case_str = json.dumps(case, ensure_ascii=False)
            reconstructed_text = reconstruct_case_deepseek(case_str)

            new_data.append({
                "case_id": case.get("CaseId"),
                "title": case.get("Case"),
                "reconstructed_analysis": reconstructed_text
            })

            time.sleep(0.5)  # 防止 API 限流

        # 保存中间结果，防止程序中途挂掉丢失
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"重构完成，已保存到 {output_file}")

# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    input_file = "0.json"  # 原始数据
    output_file = "0_reconstructed_deepseek.json"  # 重构后输出
    process_json_file_deepseek(input_file, output_file, batch_size=5)
