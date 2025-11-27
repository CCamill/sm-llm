# 使用 Qwen-Coder 生成代码 Embedding 指南

## 📋 概述

你可以使用 Qwen2.5-Coder-7B 来生成代码的 embedding 向量，但需要了解一些技术细节。

## 🔧 方案一：使用量化的 Qwen-Coder（已提供脚本）

### 显存需求估算

| 量化方式 | 显存占用 | 适用场景 |
|---------|---------|---------|
| 4-bit (NF4) | ~5-6 GB | 推荐，4090 完全够用 |
| 8-bit | ~8-10 GB | 精度更高，4090 足够 |
| 无量化 (bf16) | ~14 GB | 最佳精度，4090 可以但余量小 |

### 使用方法

```bash
# 安装依赖
pip install torch transformers accelerate bitsandbytes sentencepiece

# 运行脚本
python qwen_coder_embedding.py
```

### 关键技术点

1. **Pooling 策略**
   - `last`: 取最后一个 token 的 hidden state（最常用）
   - `mean`: 所有 token 的平均值
   - 对于 decoder-only 模型，`last` 通常效果最好

2. **L2 归一化**
   - 归一化后计算余弦相似度更方便
   - 直接使用点积即可

---

## ⚡ 方案二：更好的替代方案（推荐）

如果你的主要目标是**代码语义搜索或相似度计算**，有专门为此设计的 embedding 模型，效果更好且更轻量：

### 推荐模型

| 模型 | 大小 | 特点 |
|-----|------|-----|
| `jinaai/jina-embeddings-v3` | ~0.5B | 代码/文本通用，MTEB 高分 |
| `BAAI/bge-code-embedding-v1.5` | ~0.3B | 专为代码设计 |
| `microsoft/codebert-base` | ~125M | 微软的代码 BERT |
| `Salesforce/codet5p-110m-embedding` | ~110M | CodeT5+ 的 embedding 版本 |

### 示例：使用 sentence-transformers

```python
from sentence_transformers import SentenceTransformer

# 加载专门的代码 embedding 模型（更小更快）
model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)

# 获取 embedding
codes = [
    "def quick_sort(arr): ...",
    "function quickSort(arr) { ... }",
]
embeddings = model.encode(codes)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])
```

---

## 🤔 什么时候用 Qwen-Coder？

### 适合使用 Qwen-Coder 生成 embedding 的场景：

1. **需要理解代码上下文和意图**
   - 如判断代码功能、bug 模式等
   
2. **需要结合生成能力**
   - 如先理解代码再生成建议
   
3. **已经在用 Qwen-Coder 做其他任务**
   - 复用模型，减少资源

### 不太适合的场景：

1. **纯粹的代码检索/搜索**
   - 专门的 embedding 模型效果更好
   
2. **大规模批量处理**
   - 7B 模型太大，专用模型更高效
   
3. **需要低延迟**
   - 大模型推理慢

---

## 🚀 进阶：使用 LLM2Vec 方法

如果想让 decoder-only 模型表现得更像 encoder（更好的 embedding），可以考虑 LLM2Vec 方法：

```python
# 安装
pip install llm2vec

from llm2vec import LLM2Vec

# 这会修改模型使其支持双向注意力
model = LLM2Vec.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    peft_model_name_or_path="...",  # 需要对应的 LoRA 权重
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
```

注意：LLM2Vec 需要特定的 LoRA 权重，可能没有现成的 Qwen-Coder 版本。

---

## 📊 性能对比参考

| 方法 | embedding 质量 | 速度 | 显存 |
|-----|--------------|------|------|
| Qwen-Coder-7B (4bit) | ⭐⭐⭐ | ⭐⭐ | 5-6 GB |
| 专用 code embedding 模型 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | <2 GB |
| LLM2Vec 改造 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 5-6 GB |

---

## 总结

1. **如果你想尝试用 Qwen-Coder**：使用提供的 `qwen_coder_embedding.py` 脚本
2. **如果你追求最佳 embedding 效果**：使用专门的代码 embedding 模型
3. **4090 显存完全足够**运行 4-bit 量化的 7B 模型
