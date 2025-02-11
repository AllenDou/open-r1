from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 指定模型名称（可以换成其他 Qwen 版本，如 "Qwen/Qwen-14B"）
model_name = "data/Qwen2.5-0.5B-Open-R1-Distill"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pdb; pdb.set_trace()
# 加载模型（使用 float16 以减少显存占用）
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 生成文本示例
input_text = "你好，通义千问！"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 确保数据传输到 GPU
output = model.generate(**inputs, max_length=100)

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

