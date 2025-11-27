import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ----------------------------------------------------------------
# 1. 配置参数
# ----------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" # 或者你下载好的本地路径
NEW_MODEL_NAME = "resources/modules/Qwen2.5-Coder-7B-Assembly-LoRA"

# 24GB 显存关键设置
# 汇编代码通常比较长，但为了显存不溢出，这里建议先设为 2048 或 4096
# 如果显存够用，可以尝试调大到 8192
MAX_SEQ_LENGTH = 2048 

# ----------------------------------------------------------------
# 2. 量化配置 (4-bit loading)
# ----------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normalized Float 4，精度更高
    bnb_4bit_compute_dtype=torch.bfloat16, # 如果是 3090/4090 必选 bf16，防止溢出
    bnb_4bit_use_double_quant=True, # 二次量化，进一步节省显存
)

# ----------------------------------------------------------------
# 3. 加载模型与 Tokenizer
# ----------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen 没有默认 pad_token
tokenizer.padding_side = "right" # 训练通常用 right padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa" # 强烈建议开启 Flash Attention 2
)

# 预处理模型以进行 k-bit 训练 (开启梯度检查点等)
model = prepare_model_for_kbit_training(model)

# ----------------------------------------------------------------
# 4. LoRA 适配器配置 (针对汇编优化的架构修改)
# ----------------------------------------------------------------
peft_config = LoraConfig(
    r=64,             # Rank 较大，以适应汇编语言这种新领域
    lora_alpha=128,   # Alpha 建议是 Rank 的 2 倍
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 关键：Target Modules 设为 "all-linear"，这会覆盖 MLP 层
    # 这让 LoRA 的效果最接近全参数微调
    target_modules="all-linear" 
)

# ----------------------------------------------------------------
# 5. 加载数据集
# ----------------------------------------------------------------
# 假设你有一个 JSONL 文件: {"instruction": "...", "input": "...", "output": "..."}
# 请替换为你的实际数据路径
# dataset = load_dataset("json", data_files="path/to/your/assembly_data.jsonl", split="train")

# --- 这里为了脚本能跑，我伪造一个简单的汇编数据示例 ---
from datasets import Dataset
data = [
    {
        "messages": [
            {"role": "user", "content": "Analyze this assembly: MOV EAX, 1"},
            {"role": "assistant", "content": "This instruction moves the immediate value 1 into the EAX register."}
        ]
    }
] * 100 # 复制以模拟数据
dataset = Dataset.from_list(data)
# -----------------------------------------------------------

# ----------------------------------------------------------------
# 6. 训练参数配置
# ----------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="resources/results",
    num_train_epochs=1,
    per_device_train_batch_size=2,  # 24GB 显存建议设为 1 或 2
    gradient_accumulation_steps=8,  # 累积梯度，变相增大 Batch Size (2*8=16)
    learning_rate=2e-4,             # QLoRA 通常使用稍大的学习率
    weight_decay=0.001,
    fp16=False,
    bf16=True,                      # Ampere 架构显卡建议开启
    max_grad_norm=0.3,              # 梯度裁剪，防止梯度爆炸
    warmup_ratio=0.03,
    group_by_length=True,           # 将长度相似的样本分在一组，提高训练效率
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_32bit",      # 【关键】使用分页优化器，显存不够时借用内存
    gradient_checkpointing=True,    # 【关键】以计算换显存
    report_to="none"                # 关闭 wandb，或者填 "wandb"
)

# ----------------------------------------------------------------
# 7. 开始训练 (SFTTrainer)
# ----------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"Model saved to {NEW_MODEL_NAME}")