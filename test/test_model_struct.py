# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                             device_map="auto")
print(model)