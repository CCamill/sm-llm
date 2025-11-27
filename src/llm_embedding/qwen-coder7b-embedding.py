"""
使用量化加载的 Qwen系类llm生成代码 Embedding
适用于 RTX 4090 (24GB VRAM)
"""

import argparse
import logging
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Union, Optional, Literal
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataplay import FuncDataset


def setup_logger(name:str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class QwenCoderEmbedding:
    """
    使用 qwen系列llm生成代码 embedding 的类
    支持 4-bit 和 8-bit 量化加载
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B",
        quantization: str = "4bit",  # "4bit", "8bit", "none"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_instruction: bool = True,  # 是否使用 Instruct 版本
        cpu_offload: bool = True,
        use_flash_attention: bool = True,
        max_memory: Optional[dict] = None,
    ):
        """
        初始化模型
        
        Args:
            model_name: 模型名称或路径
            quantization: 量化方式 - "4bit", "8bit", "none"
            device: 设备映射方式
            use_instruction: 是否使用 Instruct 版本（通常更适合理解语义）
            cpu_offload: 是否启用 CPU offload（节省显存）
            use_flash_attention: 是否使用 Flash Attention（加速注意力计算）
        """
        if use_instruction and "Instruct" not in model_name:
            model_name = model_name + "-Instruct"
        
        self.model_name = model_name
        self.device = device
        
        # 配置量化参数
        quantization_config = None
        if quantization == "4bit":
            # 4-bit 量化配置 (NF4 格式，双重量化节省更多内存)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # NF4 对正态分布权重效果更好
                bnb_4bit_use_double_quant=True,  # 双重量化，额外节省 0.4 bit/参数
                bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
            )
            print(f"✓ 使用 4-bit 量化 (NF4 + 双重量化)")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print(f"✓ 使用 8-bit 量化")
        else:
            print(f"✓ 不使用量化")
        
        # 配置 attention
        attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
        
        # 配置 device_map
        if cpu_offload:
            # 自动分配，允许 CPU offload
            device_map = "auto"
            if max_memory is None:
                max_memory = {
                    0: "22GB",  # 为 4090 留一些余量
                    "cpu": "16GB",
                }
            print(f"启用 CPU Offload，内存限制: {max_memory}")
        else:
            # 非 offload 模式：优先使用GPU
            if torch.cuda.is_available():
                device_map = 0  # 使用整数索引，不是 "cuda"
                max_memory = None
            else:
                device_map = "cpu"
                max_memory = None
        
        print(f"加载模型: {model_name}...")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # 加载模型
        print(f"加载模型（{quantization} 量化）...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Flash Attention 加载失败，回退到 SDPA: {e}")
            self.model = git AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        
        # 获取 hidden size
        self.hidden_size = self.model.config.hidden_size
        print(f"✓ 模型加载完成！Embedding 维度: {self.hidden_size}")
        
        # 打印显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU 显存使用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")
    
    def get_embedding(
        self,
        texts: Union[str, List[str]],
        pooling: str = "last",  # "last", "mean", "eos"
        normalize: bool = True,
        max_length: int = 2048,
    ) -> np.ndarray:
        """
        获取文本的 embedding 向量
        
        Args:
            texts: 输入文本（单个或列表）
            pooling: 池化方式
                - "last": 使用最后一个 token 的 hidden state（推荐）
                - "mean": 对所有 token 取平均
                - "eos": 使用 EOS token 的 hidden state
            normalize: 是否 L2 归一化
            max_length: 最大 token 长度
            
        Returns:
            embedding 向量，shape: (batch_size, hidden_size)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 可选：添加任务提示（对于 Instruct 模型可能有帮助）
        # 这里不加提示，直接编码原始文本
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # 移动到设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 前向传播，获取 hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 获取最后一层的 hidden states
        # shape: (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]
        
        # 根据池化方式获取 embedding
        if pooling == "last":
            # 获取每个序列最后一个非 padding token 的位置
            attention_mask = inputs["attention_mask"]
            # 找到每个序列的最后一个有效 token 位置
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.size(0)
            embeddings = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                seq_lengths
            ]
        
        elif pooling == "mean":
            # 使用 attention mask 进行加权平均
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (last_hidden_states * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1)
        
        elif pooling == "eos":
            # 添加 EOS token 后取其 hidden state
            # 这里简化处理，使用最后一个 token
            embeddings = last_hidden_states[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # L2 归一化
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()



# ============== 使用示例 ==============

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder-7B 代码 Embedding 示例")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B", help="模型名称或路径")
    parser.add_argument('--log_path', type=str, default=f"resources/logs/Qwen2.5-Coder-7B-embedding-{int(time.time())}.log", help='Path to the log file')
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="量化方式")
    parser.add_argument("--use_instruction", action="store_true", help="是否使用 Instruct 版本")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备映射方式")
    parser.add_argument("--data_path", type=str, default="resources/datasets/dataset.csv", help="数据路径")
    parser.add_argument('--ratio_pos_neg', type=float, default=0.5, help='Ratio of positive to negative samples [0.5, 1.0]')
    parser.add_argument('--items_num', type=int, default=None, help='Number of items to load from the dataset')
    args = parser.parse_args()

    dataset = FuncDataset(args.data_path, items_num=args.items_num, ratio_pos_neg=args.ratio_pos_neg, task_type="selection")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model_list = ["Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen3-Coder-30B-A3B"]
    model_name = "Qwen/Qwen3-Coder-30B-A3B"
    cpu_offload = False
    if model_name == "Qwen/Qwen3-Coder-30B-A3B":
        cpu_offload = True

    # 1. 初始化模型（4-bit 量化，约需 5-6GB 显存）
    embedder = QwenCoderEmbedding(
        model_name=model_name,
        quantization=args.quantization,  # 4090 可以跑 4bit 或 8bit
        use_instruction=True,  # 使用 Instruct 版本
        cpu_offload=cpu_offload,
    )
    logger = setup_logger(f"{model_name.split('/')[-1]}-embedding", args.log_path)
    logger.info("==================================================================")
    logger.info(f"Log path: {args.log_path}")
    logger.info(f"Evaluating model: {model_name}")
    logger.info("量化方式: {}".format(args.quantization))
    logger.info(f"cpu_offload: {cpu_offload}")
    logger.info(f"Dataset size: {len(dataset)}")
    progress_bar = tqdm(enumerate(dataloader), 
                        total=len(dataloader),
                        desc=f"Evaluating",
                        ncols=150)
    avg = []
    gt = []
    for idx, (asm_funcs, src_funcs, labels) in progress_bar:
        anchor = []
        pos = []
        for asm_func, src_func, label in zip(asm_funcs, src_funcs, labels):
            asm_emb = embedder.get_embedding(asm_func, pooling="last")
            src_emb = embedder.get_embedding(src_func, pooling="last")
            anchor.append(asm_emb)
            pos.append(src_emb)
        
        ans = 0
        for i in range(len(anchor)):
            vA=torch.tensor(anchor[i]).cpu()
            sim=[]
            for j in range(len(pos)):
                vB=torch.tensor(pos[j]).cpu()
                AB_sim = F.cosine_similarity(vA, vB).item()
                sim.append(AB_sim)
            sim=np.array(sim)
            y=np.argsort(-sim)
            posi = 0
            for j in range(len(pos)):
                if y[j]==i:
                    posi=j+1
                    break
            
            gt.append(sim[i])

            ans += 1/posi
        ans = ans /len(anchor)
        avg.append(ans)
    logger.info(f"Mean Reciprocal Rank (MRR): {np.mean(np.array(avg))}")


if __name__ == "__main__":
    main()