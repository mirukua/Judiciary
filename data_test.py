import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# -------------------------------
# 模型加载
# -------------------------------
model_name = "Qwen/Qwen-7B-Chat"  # 可换为你本地 Qwen 模型路径
device = "cuda" if torch.cuda.is_available() else "cpu"

print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
model.eval()
print("模型加载完成。")

# -------------------------------
# 重构函数
# -------------------------------
def reconstruct_case_qwen(case_fact):
    """
    使用 Qwen 模型将原始案件事实重构为法律三段论格式
    """
    prompt = f"""
请根据以下案件事实或题目，按照法律三段论（大前提—小前提—结论）进行重构，生成高质量中文回答：
案件事实 / 题目：
{case_fact}

要求：
- 大前提：引用适用法律条文或原则
- 小前提：结合案件事实分析
- 结论：给出裁判结果或答案
- 使用中文
请严格按照以下格式输出：
大前提（适用法律）：
小前提（案件事实）：
结论（裁判结果 / 答案）：
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.2,
            do_sample=False
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()

# -------------------------------
# 读取原始 JSON 并处理
# -------------------------------
def process_json_file(input_file, output_file):
    """
    读取原始 JSON 文件，对每条案件或题目数据重构
    输出新的 JSON 文件：包含 input-output 对
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for idx, item in enumerate(data):
        # 尝试从 fact 或 question 字段读取原始内容
        case_fact = item.get("fact") or item.get("question") or item.get("query")
        if not case_fact:
            continue

        print(f"Processing item {idx+1}/{len(data)} ...")
        try:
            output = reconstruct_case_qwen(case_fact)
            new_data.append({
                "input": case_fact,
                "output": output
            })
        except Exception as e:
            print(f"第 {idx+1} 条处理失败:", e)
            continue

        # 控制推理速度，避免显存压力
        time.sleep(0.5)

    # 保存到新文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"重构完成，已保存到 {output_file}")

# -------------------------------
# 主函数
# -------------------------------
if __name__ == "__main__":
    input_file = "0.json"  # 原始数据
    output_file = "0_reconstructed.json"  # 重构后输出
    process_json_file(input_file, output_file)
