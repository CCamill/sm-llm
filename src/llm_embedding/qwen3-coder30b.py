"""
使用量化加载的 Qwen3-Coder-30B 生成代码 Embedding
适用于 RTX 4090 (24GB VRAM)

模型选择说明：
- Qwen/Qwen3-32B: Qwen3 基础模型 32B
- Qwen/Qwen3-30B-A3B: Qwen3 MoE 模型（30B 总参数，3B 激活参数）
- Qwen/Qwen2.5-Coder-32B-Instruct: Qwen2.5 代码专用模型 32B

安装依赖：
pip install torch transformers accelerate bitsandbytes sentencepiece --break-system-packages

对于 30B+ 模型，4-bit 量化后约需 15-18GB 显存
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Union, Optional, Literal
import numpy as np
import gc


class Qwen3CoderEmbedding:
    """
    使用 Qwen3-Coder 或其他大型 Qwen 模型生成代码 embedding
    支持 4-bit 量化 + CPU Offload 以适应有限显存
    """
    
    # 常用模型列表
    AVAILABLE_MODELS = {
        # Qwen3 系列
        "qwen3-32b": "Qwen/Qwen3-32B",
        "qwen3-32b-instruct": "Qwen/Qwen3-32B-Instruct",
        "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",  # MoE 模型
        "qwen3-30b-a3b-instruct": "Qwen/Qwen3-30B-A3B-Instruct",
        
        # Qwen2.5-Coder 系列（代码专用）
        "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B",
        "qwen2.5-coder-32b-instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen2.5-coder-14b-instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization: Literal["4bit", "8bit", "none"] = "4bit",
        use_flash_attention: bool = True,
        max_memory: Optional[dict] = None,
        cpu_offload: bool = False,
    ):
        """
        初始化模型
        
        Args:
            model_name: 模型名称，可以是完整路径或简称（见 AVAILABLE_MODELS）
            quantization: 量化方式
            use_flash_attention: 是否使用 Flash Attention 2（需要安装 flash-attn）
            max_memory: 显存限制，如 {"cuda:0": "20GB", "cpu": "30GB"}
            cpu_offload: 是否启用 CPU offload（显存不足时使用）
        """
        # 解析模型名称
        if model_name.lower() in self.AVAILABLE_MODELS:
            model_name = self.AVAILABLE_MODELS[model_name.lower()]
        
        self.model_name = model_name
        print(f"="*60)
        print(f"初始化模型: {model_name}")
        print(f"="*60)
        
        # 配置量化
        quantization_config = self._get_quantization_config(quantization)
        
        # 配置 attention
        attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
        
        # 加载 tokenizer
        print("加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # 对于 decoder-only 模型，左填充更好
        )
        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置 device_map
        if cpu_offload:
            # 自动分配，允许 CPU offload
            device_map = "auto"
            if max_memory is None:
                max_memory = {
                    "cuda": "22GB",  # 为 4090 留一些余量
                    "cpu": "48GB",
                }
            print(f"启用 CPU Offload，内存限制: {max_memory}")
        else:
            device_map = "auto"
        
        # 加载模型
        print(f"加载模型（{quantization} 量化）...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Flash Attention 加载失败，回退到 SDPA: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        
        # 获取模型信息
        self.hidden_size = self.model.config.hidden_size
        self._print_model_info()
    
    def _get_quantization_config(self, quantization: str) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if quantization == "4bit":
            print("✓ 使用 4-bit NF4 量化 + 双重量化")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            print("✓ 使用 8-bit 量化")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            print("✓ 不使用量化（需要大量显存）")
            return None
    
    def _print_model_info(self):
        """打印模型信息"""
        print(f"\n{'='*60}")
        print(f"模型加载完成！")
        print(f"{'='*60}")
        print(f"  Embedding 维度: {self.hidden_size}")
        print(f"  模型层数: {self.model.config.num_hidden_layers}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU 显存: {allocated:.2f}GB / {total:.2f}GB")
        
        # 检查模型分布
        if hasattr(self.model, 'hf_device_map'):
            devices = set(self.model.hf_device_map.values())
            print(f"  模型分布: {devices}")
        print(f"{'='*60}\n")
    
    def get_embedding(
        self,
        texts: Union[str, List[str]],
        pooling: Literal["last", "mean", "weighted_mean"] = "last",
        normalize: bool = True,
        max_length: int = 4096,
        batch_size: int = 1,
        layer: int = -1,  # -1 表示最后一层
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        获取文本的 embedding 向量
        
        Args:
            texts: 输入文本
            pooling: 池化方式
                - "last": 最后一个 token（推荐）
                - "mean": 平均池化
                - "weighted_mean": 位置加权平均（后面的 token 权重更高）
            normalize: L2 归一化
            max_length: 最大 token 长度
            batch_size: 批处理大小（大模型建议设为 1）
            layer: 使用第几层的 hidden state，-1 表示最后一层
            show_progress: 显示进度
            
        Returns:
            embedding 向量，shape: (batch_size, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress and len(texts) > batch_size:
                print(f"  处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            # 移动到模型设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # 获取指定层的 hidden states
            hidden_states = outputs.hidden_states[layer]
            attention_mask = inputs["attention_mask"]
            
            # 池化
            embeddings = self._pool(hidden_states, attention_mask, pooling)
            
            # 归一化
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            # 清理显存
            del outputs, hidden_states
            torch.cuda.empty_cache()
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        """池化操作"""
        if pooling == "last":
            # 获取每个序列最后一个有效 token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths
            ]
        
        elif pooling == "mean":
            # 平均池化（排除 padding）
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        
        elif pooling == "weighted_mean":
            # 位置加权平均（后面的 token 权重更高）
            batch_size, seq_len, _ = hidden_states.shape
            weights = torch.arange(1, seq_len + 1, device=hidden_states.device).float()
            weights = weights.unsqueeze(0).expand(batch_size, -1)
            weights = weights * attention_mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True)
            embeddings = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        return embeddings
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        **kwargs,
    ) -> float:
        """计算两段文本的余弦相似度"""
        emb1 = self.get_embedding(text1, show_progress=False, **kwargs)
        emb2 = self.get_embedding(text2, show_progress=False, **kwargs)
        return float(np.dot(emb1[0], emb2[0]))
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        **kwargs,
    ) -> List[tuple]:
        """在候选列表中找到最相似的文本"""
        query_emb = self.get_embedding(query, show_progress=False, **kwargs)
        candidate_embs = self.get_embedding(candidates, **kwargs)
        
        # 计算相似度
        similarities = np.dot(candidate_embs, query_emb[0])
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(candidates[i], float(similarities[i])) for i in top_indices]


def estimate_memory(model_params_b: float, quantization: str) -> float:
    """估算显存需求（GB）"""
    if quantization == "4bit":
        return model_params_b * 0.5 + 2  # 4-bit ≈ 0.5 bytes/param + overhead
    elif quantization == "8bit":
        return model_params_b * 1.0 + 2
    else:
        return model_params_b * 2.0 + 2  # bf16 = 2 bytes/param


def main():
    print("\n" + "="*60)
    print("Qwen3/Qwen2.5-Coder 大模型 Embedding 示例")
    print("="*60)
    
    # 显存估算
    print("\n📊 显存需求估算 (4-bit 量化):")
    print(f"  Qwen3-32B:           ~{estimate_memory(32, '4bit'):.1f} GB")
    print(f"  Qwen3-30B-A3B (MoE): ~{estimate_memory(30, '4bit'):.1f} GB (实际更低)")
    print(f"  Qwen2.5-Coder-32B:   ~{estimate_memory(32, '4bit'):.1f} GB")
    print(f"  RTX 4090 显存:       24 GB")
    
    # 选择模型
    # 对于 4090，推荐以下选项：
    # 1. "qwen2.5-coder-32b-instruct" - 代码专用，效果最好
    # 2. "qwen3-30b-a3b-instruct" - MoE 模型，实际激活参数少
    # 3. "qwen3-32b-instruct" - 通用模型
    
    print("\n🚀 加载模型...")
    
    # 使用 Qwen2.5-Coder-32B（代码专用，推荐）
    # 如果显存不足，会自动 offload 到 CPU
    embedder = Qwen3CoderEmbedding(
        model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",  # 或 "qwen3-32b-instruct"
        quantization="4bit",
        use_flash_attention=True,
        cpu_offload=True,  # 显存不足时启用
        max_memory={
            "cuda:0": "22GB",  # 为 4090 留余量
            "cpu": "64GB",     # 根据你的内存调整
        },
    )
    
    # 测试代码示例
    code_samples = [
        # 1. Python 快速排序
        '''def quick_sort(arr):
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)''',
        
        # 2. Python 归并排序
        '''def merge_sort(arr):
    """Merge sort implementation"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)''',
        
        # 3. Python HTTP 请求
        '''import requests

def fetch_data(url):
    """Fetch data from URL"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()''',
        
        # 4. JavaScript 快速排序
        '''function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    return [...quickSort(left), ...middle, ...quickSort(right)];
}''',
        
        # 5. Rust 快速排序
        '''fn quick_sort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {
    if arr.len() <= 1 {
        return arr.to_vec();
    }
    let pivot = arr[arr.len() / 2].clone();
    let left: Vec<_> = arr.iter().filter(|&x| x < &pivot).cloned().collect();
    let middle: Vec<_> = arr.iter().filter(|&x| x == &pivot).cloned().collect();
    let right: Vec<_> = arr.iter().filter(|&x| x > &pivot).cloned().collect();
    [quick_sort(&left), middle, quick_sort(&right)].concat()
}''',
    ]
    
    labels = [
        "Python QuickSort",
        "Python MergeSort", 
        "Python HTTP",
        "JS QuickSort",
        "Rust QuickSort",
    ]
    
    # 生成 embeddings
    print("\n📝 生成代码 Embeddings...")
    embeddings = embedder.get_embedding(
        code_samples,
        pooling="last",
        batch_size=1,  # 大模型用小 batch
    )
    print(f"Embedding shape: {embeddings.shape}")
    
    # 计算相似度矩阵
    print("\n" + "="*60)
    print("代码相似度矩阵（余弦相似度）:")
    print("="*60)
    
    # 表头
    print(f"\n{'':18s}", end="")
    for label in labels:
        print(f"{label:15s}", end="")
    print()
    print("-" * (18 + 15 * len(labels)))
    
    # 相似度矩阵
    for i, label_i in enumerate(labels):
        print(f"{label_i:18s}", end="")
        for j in range(len(labels)):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            print(f"{sim:15.3f}", end="")
        print()
    
    # 分析
    print("\n" + "="*60)
    print("📊 相似度分析:")
    print("="*60)
    
    pairs = [
        (0, 3, "Python vs JS QuickSort", "相同算法，不同语言"),
        (0, 4, "Python vs Rust QuickSort", "相同算法，不同语言"),
        (0, 1, "QuickSort vs MergeSort", "不同排序算法"),
        (0, 2, "QuickSort vs HTTP", "完全不同功能"),
        (3, 4, "JS vs Rust QuickSort", "相同算法，不同语言"),
    ]
    
    for i, j, name, desc in pairs:
        sim = float(np.dot(embeddings[i], embeddings[j]))
        print(f"  {name}: {sim:.4f}")
        print(f"    → {desc}")
    
    # 语义搜索示例
    print("\n" + "="*60)
    print("🔍 语义搜索示例:")
    print("="*60)
    
    query = "sorting algorithm implementation"
    print(f"\n查询: '{query}'")
    print("\n最相似的代码片段:")
    
    results = embedder.find_most_similar(query, code_samples, top_k=3)
    for i, (code, sim) in enumerate(results, 1):
        preview = code[:50].replace('\n', ' ') + "..."
        print(f"  {i}. [{sim:.4f}] {preview}")
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()