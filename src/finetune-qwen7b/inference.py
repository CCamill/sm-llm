"""
推理和嵌入提取模块

用于：
1. 加载微调后的模型
2. 提取汇编代码和源码的嵌入
3. 计算相似度和检索
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
import time

from playdata import FuncDataset

def setup_logger(name: str, log_path: str) -> logging.Logger:
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

logger = setup_logger(name = 'inference', log_path=f"resources/logs/Qwen-inference-{time.time()}.log")


@dataclass
class InferenceConfig:
    """推理配置"""
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    lora_path: str = "resources/finetune_modules/best/lora_weights"
    projection_head_path: str = "resources/finetune_modules/best/projection_head.pt"
    max_seq_length: int = 512
    batch_size: int = 32
    use_projection_head: bool = True
    pooling_strategy: str = 'mean'  # 'last_token', 'mean', 'eos'
    normalize_embeddings: bool = True
    fp16: bool = True,
    ratio_pos_neg: int=0.5,
    items_num: int=None,
    data_path:str ="resources/datasets/dataset.csv"


class ProjectionHead(torch.nn.Module):
    """投影头"""
    
    def __init__(self, hidden_size: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, projection_dim),
            torch.nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class BinarySourceEmbedder:
    """
    二进制-源码嵌入器
    
    用于提取汇编代码和源码的语义嵌入
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 提示模板
        self.asm_prompt_template = (
            "Analyze the following assembly code and understand its semantic meaning:\n"
            "```asm\n{code}\n```\n"
            "Semantic representation:"
        )
        self.source_prompt_template = (
            "Analyze the following source code and understand its semantic meaning:\n"
            "```c\n{code}\n```\n"
            "Semantic representation:"
        )
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16 if self.config.fp16 else torch.float32,
            trust_remote_code=True,
            device_map='auto'
        )
        
        # 加载LoRA权重
        if os.path.exists(self.config.lora_path):
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.lora_path
            )
            self.model = self.model.merge_and_unload()
        
        # 加载投影头
        self.projection_head = None
        if self.config.use_projection_head and os.path.exists(self.config.projection_head_path):
            hidden_size = self.model.config.hidden_size
            self.projection_head = ProjectionHead(hidden_size)
            self.projection_head.load_state_dict(
                torch.load(self.config.projection_head_path, map_location=self.device)
            )
            self.projection_head = self.projection_head.to(self.device)
            self.projection_head.eval()
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """获取嵌入"""
        with torch.no_grad():
            if self.config.fp16:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states[-1]
            
            if self.config.pooling_strategy == 'last_token':
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                embeddings = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    seq_lengths
                ]
            elif self.config.pooling_strategy == 'mean':
                mask = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (hidden_states * mask).sum(dim=1)
                count = mask.sum(dim=1)
                embeddings = sum_embeddings / count
            else:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                embeddings = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    seq_lengths
                ]
            
            # 应用投影头
            if self.projection_head is not None:
                embeddings = self.projection_head(embeddings.float())
            
            # 归一化
            if self.config.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings
    
    def encode_assembly(
        self,
        asm_codes: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        编码汇编代码
        
        Args:
            asm_codes: 单个或多个汇编代码字符串
            batch_size: 批次大小
        
        Returns:
            embeddings: 嵌入数组 [num_samples, embedding_dim]
        """
        if isinstance(asm_codes, str):
            asm_codes = [asm_codes]
        
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        for i in range(0, len(asm_codes), batch_size):
            batch = asm_codes[i:i + batch_size]
            texts = [self.asm_prompt_template.format(code=code) for code in batch]
            
            encoding = self.tokenizer(
                texts,
                max_length=self.config.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            embeddings = self._get_embedding(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def encode_source(
        self,
        source_codes: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        编码源代码
        
        Args:
            source_codes: 单个或多个源代码字符串
            batch_size: 批次大小
        
        Returns:
            embeddings: 嵌入数组 [num_samples, embedding_dim]
        """
        if isinstance(source_codes, str):
            source_codes = [source_codes]
        
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        for i in tqdm(range(0, len(source_codes), batch_size), desc="Encoding source"):
            batch = source_codes[i:i + batch_size]
            texts = [self.source_prompt_template.format(code=code) for code in batch]
            
            encoding = self.tokenizer(
                texts,
                max_length=self.config.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            embeddings = self._get_embedding(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        key_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算相似度矩阵
        
        Args:
            query_embeddings: 查询嵌入 [num_queries, embedding_dim]
            key_embeddings: 键嵌入 [num_keys, embedding_dim]
        
        Returns:
            similarity_matrix: 相似度矩阵 [num_queries, num_keys]
        """
        # 确保已归一化
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        key_embeddings = key_embeddings / np.linalg.norm(key_embeddings, axis=1, keepdims=True)
        
        return np.dot(query_embeddings, key_embeddings.T)
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        检索最相似的候选
        
        Args:
            query_embedding: 查询嵌入 [1, embedding_dim] 或 [embedding_dim]
            candidate_embeddings: 候选嵌入 [num_candidates, embedding_dim]
            top_k: 返回前k个结果
        
        Returns:
            indices: 最相似候选的索引
            scores: 相似度分数
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)[0]
        
        # 按相似度降序排列
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores


class SimilarityEvaluator:
    """
    相似度评估器
    
    用于评估模型的检索性能
    """
    
    def __init__(self, embedder: BinarySourceEmbedder):
        self.embedder = embedder
    
    def evaluate_retrieval(
        self,
        asm_codes: List[str],
        source_codes: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        评估检索性能
        
        假设asm_codes[i]和source_codes[i]是配对的
        
        Args:
            asm_codes: 汇编代码列表
            source_codes: 源代码列表
            batch_size: 批次大小
        
        Returns:
            metrics: 评估指标
        """
        logger.info("Encoding assembly codes...")
        asm_embeddings = self.embedder.encode_assembly(asm_codes, batch_size)
        
        logger.info("Encoding source codes...")
        source_embeddings = self.embedder.encode_source(source_codes, batch_size)
        
        # 计算相似度矩阵
        similarity_matrix = self.embedder.compute_similarity(asm_embeddings, source_embeddings)
        
        # 计算指标
        num_samples = len(asm_codes)
        ranks = []
        
        for i in range(num_samples):
            sims = similarity_matrix[i]
            sorted_indices = np.argsort(sims)[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        metrics = {
            'mrr': float(np.mean(1.0 / ranks)),
            'recall@1': float(np.mean(ranks <= 1)),
            'recall@5': float(np.mean(ranks <= 5)),
            'recall@10': float(np.mean(ranks <= 10)),
            'recall@100': float(np.mean(ranks <= 100)),
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks))
        }
        
        return metrics
    
    def evaluate_pairwise(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        评估配对相似度
        
        Args:
            positive_pairs: 正样本对列表 [(asm, source), ...]
            negative_pairs: 负样本对列表 [(asm, source), ...]
            batch_size: 批次大小
        
        Returns:
            metrics: 评估指标
        """
        # 处理正样本
        pos_asm = [p[0] for p in positive_pairs]
        pos_src = [p[1] for p in positive_pairs]
        pos_asm_emb = self.embedder.encode_assembly(pos_asm, batch_size)
        pos_src_emb = self.embedder.encode_source(pos_src, batch_size)
        pos_sims = np.sum(pos_asm_emb * pos_src_emb, axis=1)  # 点积（已归一化等于余弦相似度）
        
        # 处理负样本
        neg_asm = [p[0] for p in negative_pairs]
        neg_src = [p[1] for p in negative_pairs]
        neg_asm_emb = self.embedder.encode_assembly(neg_asm, batch_size)
        neg_src_emb = self.embedder.encode_source(neg_src, batch_size)
        neg_sims = np.sum(neg_asm_emb * neg_src_emb, axis=1)
        
        # 计算指标
        all_sims = np.concatenate([pos_sims, neg_sims])
        all_labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
        
        # 找最佳阈值
        thresholds = np.linspace(all_sims.min(), all_sims.max(), 100)
        best_acc = 0
        best_threshold = 0
        
        for threshold in thresholds:
            preds = (all_sims >= threshold).astype(int)
            acc = np.mean(preds == all_labels)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # 使用最佳阈值计算指标
        preds = (all_sims >= best_threshold).astype(int)
        
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': best_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_threshold': best_threshold,
            'pos_sim_mean': float(np.mean(pos_sims)),
            'pos_sim_std': float(np.std(pos_sims)),
            'neg_sim_mean': float(np.mean(neg_sims)),
            'neg_sim_std': float(np.std(neg_sims)),
            'sim_gap': float(np.mean(pos_sims) - np.mean(neg_sims))
        }
        
        return metrics


def main():
    """演示使用方法"""
    config = InferenceConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        lora_path="resources/finetune_modules/best/lora_weights",
        projection_head_path="resources/finetune_modules/best/projection_head.pt",
        use_projection_head=True,
        batch_size=32,
        ratio_pos_neg=0.5,
        items_num=None,
        data_path="resources/datasets/test_dataset.csv",
    )
    parser = argparse.ArgumentParser(description="QLora Moder inference")
    parser.add_argument("--finetune_layers", type=int, default=4)

    args = parser.parse_args()
    logger.info(f"data_path: {config.data_path}")
    logger.info(f"model name: {config.model_name} QLora")
    logger.info(f"fine-tuned layers: {args.finetune_layers}")
    logger.info(f"ratio_pos_neg: {config.ratio_pos_neg}")
    logger.info(f"batch_size: {config.batch_size}")
    logger.info(f"pooling_strategy: {config.pooling_strategy}")
    
    embedder = BinarySourceEmbedder(config)

    dataset = FuncDataset(config.data_path, items_num=config.items_num, ratio_pos_neg=config.ratio_pos_neg, task_type="selection")
    logger.info(f"asm-src func pairs: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
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
            asm_emb = embedder.encode_assembly(asm_func)
            src_emb = embedder.encode_assembly(src_func)
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

    
if __name__ == '__main__':
    main()