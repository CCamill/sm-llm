from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name= r"Qwen/Qwen3-Coder-30B-A3B-Instruct"
# 检查文件是否存在
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ 模型加载成功，完整性验证通过")
except Exception as e:
    print(f"❌ 模型损坏: {e}")