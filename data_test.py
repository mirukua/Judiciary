import json
input_file="0.json"
with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
new_data = []
ctxs = data.get("ctxs", {})

for key, case in ctxs.items():
    case_str = json.dumps(case, ensure_ascii=False)  # 直接将单条案件转换成 JSON 字符串
    # reconstructed_text = reconstruct_case_qwen(case_str)
    
    new_data.append({
        "case_id": case.get("CaseId"),
        "title": case.get("Case"),
        # "reconstructed_analysis": reconstructed_text
    })
print(new_data)
