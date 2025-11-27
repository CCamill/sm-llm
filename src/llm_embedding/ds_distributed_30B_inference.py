"""
分布式 Qwen3-Coder 30B Embedding 推理
使用 DeepSpeed 张量并行在多节点上运行

项目结构:
distributed_embedding/
├── config/
│   ├── ds_config.json          # DeepSpeed 配置
│   └── hostfile                 # 节点配置
├── src/
│   ├── __init__.py
│   ├── model_loader.py         # 模型加载
│   ├── embedding_engine.py     # Embedding 引擎
│   └── server.py               # API 服务器（可选）
├── scripts/
│   ├── setup_node.sh           # 节点环境配置
│   ├── start_master.sh         # 主节点启动
│   └── run_distributed.sh      # 分布式启动
├── requirements.txt
├── run_inference.py            # 主入口
└── README.md
"""

import os
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Union, Optional
import numpy as np
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s'
)


class DistributedQwenEmbedding:
    """
    使用 DeepSpeed 张量并行的分布式 Embedding 引擎
    支持多节点多 GPU 推理
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        ds_config_path: str = "config/ds_config.json",
        local_rank: int = -1,
        world_size: int = 2,
    ):
        """
        初始化分布式模型
        
        Args:
            model_name: 模型名称或路径
            ds_config_path: DeepSpeed 配置文件路径
            local_rank: 本地 GPU rank
            world_size: 总 GPU 数量
        """
        self.model_name = model_name
        self.local_rank = local_rank if local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = world_size
        
        # 获取全局 rank
        self.global_rank = int(os.environ.get("RANK", 0))
        
        self.logger = logging.getLogger(__name__)
        self.logger = logging.LoggerAdapter(self.logger, {'rank': self.global_rank})
        
        self.logger.info(f"初始化分布式环境: local_rank={self.local_rank}, global_rank={self.global_rank}")
        
        # 加载 DeepSpeed 配置
        with open(ds_config_path, 'r') as f:
            self.ds_config = json.load(f)
        
        # 更新张量并行大小
        self.ds_config["tensor_parallel"]["tp_size"] = world_size
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化分布式模型"""
        self.logger.info(f"加载模型: {self.model_name}")
        
        # 加载 tokenizer（所有 rank 都需要）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型配置
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.hidden_size = config.hidden_size
        
        # 使用 DeepSpeed Inference 加载模型
        # 这会自动进行张量并行分割
        self.logger.info("使用 DeepSpeed Inference 初始化模型...")
        
        # 先加载模型到 CPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # 使用 DeepSpeed 初始化
        self.model = deepspeed.init_inference(
            model,
            mp_size=self.world_size,  # 模型并行大小
            dtype=torch.float16,
            replace_with_kernel_inject=True,  # 使用优化的 kernel
            replace_method="auto",
        )
        
        self.model.eval()
        self.logger.info(f"模型加载完成！Hidden size: {self.hidden_size}")
        
        # 打印显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"GPU {self.local_rank} 显存使用: {allocated:.2f} GB")
    
    def get_embedding(
        self,
        texts: Union[str, List[str]],
        pooling: str = "last",
        normalize: bool = True,
        max_length: int = 4096,
    ) -> Optional[np.ndarray]:
        """
        获取文本 embedding（仅在 rank 0 返回结果）
        
        Args:
            texts: 输入文本
            pooling: 池化方式 ("last", "mean")
            normalize: 是否 L2 归一化
            max_length: 最大 token 长度
            
        Returns:
            embedding 向量（仅 rank 0 返回，其他 rank 返回 None）
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 移动到当前 GPU
        inputs = {k: v.cuda(self.local_rank) for k, v in inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 获取最后一层 hidden states
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        
        # 池化
        if pooling == "last":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths
            ]
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # 归一化
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 只在 rank 0 收集并返回结果
        if self.global_rank == 0:
            return embeddings.cpu().numpy()
        return None
    
    def compute_similarity(self, text1: str, text2: str) -> Optional[float]:
        """计算两段文本的余弦相似度"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if self.global_rank == 0 and emb1 is not None and emb2 is not None:
            return float(np.dot(emb1[0], emb2[0]))
        return None


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--config", type=str, default="ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # DeepSpeed 会设置这些环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    
    # 初始化分布式环境
    deepspeed.init_distributed()
    
    # 创建 embedding 引擎
    engine = DistributedQwenEmbedding(
        model_name=args.model,
        ds_config_path=args.config,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    # 测试
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        print("\n" + "="*60)
        print("分布式 Embedding 测试")
        print("="*60)
    
    # 同步
    dist.barrier()
    
    # 测试代码
    test_codes = [
        """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
        
        """function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    return [...quickSort(left), ...middle, ...quickSort(right)];
}""",
    ]
    
    embeddings = engine.get_embedding(test_codes)
    
    if rank == 0 and embeddings is not None:
        print(f"\nEmbedding shape: {embeddings.shape}")
        sim = float(np.dot(embeddings[0], embeddings[1]))
        print(f"Python vs JS QuickSort 相似度: {sim:.4f}")
        print("\n✅ 分布式推理成功！")
    
    dist.barrier()


if __name__ == "__main__":
    main()
