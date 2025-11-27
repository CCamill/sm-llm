"""
高级训练策略

包含:
1. 困难负样本挖掘 (Hard Negative Mining)
2. 课程学习 (Curriculum Learning)
3. 多阶段训练 (Multi-stage Training)
4. 动量对比学习 (Momentum Contrastive Learning - MoCo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)


class MomentumQueue:
    """
    动量队列 - 用于MoCo风格的对比学习
    
    维护一个大型的负样本队列，无需大batch size即可获得大量负样本
    """
    
    def __init__(self, feature_dim: int, queue_size: int = 65536):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        
        # 初始化队列
        self.queue = torch.randn(feature_dim, queue_size)
        self.queue = F.normalize(self.queue, dim=0)
        
        self.ptr = 0
    
    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor):
        """将新的嵌入加入队列"""
        batch_size = embeddings.shape[0]
        
        # 如果队列满了，覆盖最旧的
        if self.ptr + batch_size > self.queue_size:
            # 分两部分插入
            remaining = self.queue_size - self.ptr
            self.queue[:, self.ptr:] = embeddings[:remaining].T
            self.queue[:, :batch_size - remaining] = embeddings[remaining:].T
            self.ptr = batch_size - remaining
        else:
            self.queue[:, self.ptr:self.ptr + batch_size] = embeddings.T
            self.ptr += batch_size
    
    def get_queue(self) -> torch.Tensor:
        """获取队列中的所有嵌入"""
        return self.queue.clone()


class MoCoLoss(nn.Module):
    """
    Momentum Contrast (MoCo) 损失
    
    使用动量编码器和大型负样本队列实现高效的对比学习
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        queue_size: int = 65536,
        momentum: float = 0.999
    ):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum
        
        self.asm_queue = None
        self.source_queue = None
    
    def init_queues(self, feature_dim: int, device: torch.device):
        """初始化队列"""
        self.asm_queue = MomentumQueue(feature_dim, self.queue_size)
        self.asm_queue.queue = self.asm_queue.queue.to(device)
        
        self.source_queue = MomentumQueue(feature_dim, self.queue_size)
        self.source_queue.queue = self.source_queue.queue.to(device)
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        asm_embeddings_momentum: torch.Tensor,
        source_embeddings_momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算MoCo损失
        
        Args:
            asm_embeddings: 汇编嵌入 (query encoder)
            source_embeddings: 源码嵌入 (query encoder)
            asm_embeddings_momentum: 汇编嵌入 (momentum encoder)
            source_embeddings_momentum: 源码嵌入 (momentum encoder)
        """
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device
        
        # 初始化队列
        if self.asm_queue is None:
            self.init_queues(asm_embeddings.shape[1], device)
        
        # L2归一化
        asm_embeddings = F.normalize(asm_embeddings, dim=1)
        source_embeddings = F.normalize(source_embeddings, dim=1)
        asm_embeddings_momentum = F.normalize(asm_embeddings_momentum, dim=1)
        source_embeddings_momentum = F.normalize(source_embeddings_momentum, dim=1)
        
        # 获取队列中的负样本
        source_queue = self.source_queue.get_queue().to(device)
        asm_queue = self.asm_queue.get_queue().to(device)
        
        # 计算 asm -> source 的损失
        # 正样本：当前batch中配对的source
        l_pos_a2s = torch.einsum('nc,nc->n', [asm_embeddings, source_embeddings_momentum]).unsqueeze(-1)
        # 负样本：队列中的source
        l_neg_a2s = torch.einsum('nc,ck->nk', [asm_embeddings, source_queue])
        logits_a2s = torch.cat([l_pos_a2s, l_neg_a2s], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss_a2s = F.cross_entropy(logits_a2s, labels)
        
        # 计算 source -> asm 的损失
        l_pos_s2a = torch.einsum('nc,nc->n', [source_embeddings, asm_embeddings_momentum]).unsqueeze(-1)
        l_neg_s2a = torch.einsum('nc,ck->nk', [source_embeddings, asm_queue])
        logits_s2a = torch.cat([l_pos_s2a, l_neg_s2a], dim=1) / self.temperature
        loss_s2a = F.cross_entropy(logits_s2a, labels)
        
        total_loss = (loss_a2s + loss_s2a) / 2
        
        # 更新队列（使用动量编码器的输出）
        self.source_queue.enqueue(source_embeddings_momentum.detach())
        self.asm_queue.enqueue(asm_embeddings_momentum.detach())
        
        # 计算指标
        with torch.no_grad():
            acc_a2s = (logits_a2s.argmax(dim=1) == 0).float().mean().item()
            acc_s2a = (logits_s2a.argmax(dim=1) == 0).float().mean().item()
        
        metrics = {
            'accuracy_a2s': acc_a2s,
            'accuracy_s2a': acc_s2a,
            'loss_a2s': loss_a2s.item(),
            'loss_s2a': loss_s2a.item()
        }
        
        return total_loss, metrics


class HardNegativeMiner:
    """
    困难负样本挖掘器
    
    使用预计算的嵌入来找到困难负样本
    """
    
    def __init__(
        self,
        embedder,
        num_hard_negatives: int = 5,
        refresh_interval: int = 1000
    ):
        self.embedder = embedder
        self.num_hard_negatives = num_hard_negatives
        self.refresh_interval = refresh_interval
        
        self.asm_embeddings_cache = None
        self.source_embeddings_cache = None
        self.step_counter = 0
    
    def refresh_cache(self, dataset: Dataset):
        """刷新嵌入缓存"""
        logger.info("Refreshing embedding cache for hard negative mining...")
        
        asm_codes = [dataset[i]['asm_code'] for i in range(len(dataset))]
        source_codes = [dataset[i]['source_code'] for i in range(len(dataset))]
        
        self.asm_embeddings_cache = self.embedder.encode_assembly(asm_codes)
        self.source_embeddings_cache = self.embedder.encode_source(source_codes)
        
        logger.info("Embedding cache refreshed")
    
    def mine_hard_negatives(
        self,
        query_indices: List[int],
        query_type: str = 'asm'  # 'asm' or 'source'
    ) -> List[List[int]]:
        """
        为每个查询挖掘困难负样本
        
        Args:
            query_indices: 查询样本的索引
            query_type: 查询类型
        
        Returns:
            hard_negative_indices: 每个查询对应的困难负样本索引列表
        """
        if self.asm_embeddings_cache is None:
            raise ValueError("Please refresh cache first")
        
        if query_type == 'asm':
            query_embeddings = self.asm_embeddings_cache[query_indices]
            key_embeddings = self.source_embeddings_cache
        else:
            query_embeddings = self.source_embeddings_cache[query_indices]
            key_embeddings = self.asm_embeddings_cache
        
        # 计算相似度
        similarities = np.dot(query_embeddings, key_embeddings.T)
        
        hard_negatives = []
        for i, idx in enumerate(query_indices):
            sims = similarities[i].copy()
            # 排除自己（正样本）
            sims[idx] = -float('inf')
            # 选择最相似的k个作为困难负样本
            hard_neg_idx = np.argsort(sims)[::-1][:self.num_hard_negatives]
            hard_negatives.append(hard_neg_idx.tolist())
        
        return hard_negatives


class CurriculumSampler(Sampler):
    """
    课程学习采样器
    
    从简单样本开始，逐渐引入困难样本
    """
    
    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: List[float],
        num_epochs: int,
        curriculum_type: str = 'linear'  # 'linear', 'sqrt', 'step'
    ):
        self.dataset = dataset
        self.difficulty_scores = np.array(difficulty_scores)
        self.num_epochs = num_epochs
        self.curriculum_type = curriculum_type
        
        # 按难度排序
        self.sorted_indices = np.argsort(self.difficulty_scores)
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def __iter__(self):
        # 计算当前应该包含的样本比例
        progress = (self.current_epoch + 1) / self.num_epochs
        
        if self.curriculum_type == 'linear':
            ratio = progress
        elif self.curriculum_type == 'sqrt':
            ratio = np.sqrt(progress)
        elif self.curriculum_type == 'step':
            ratio = min(1.0, (self.current_epoch // 3 + 1) * 0.25)
        else:
            ratio = 1.0
        
        # 选择前ratio比例的样本（按难度排序）
        num_samples = max(1, int(len(self.dataset) * ratio))
        available_indices = self.sorted_indices[:num_samples]
        
        # 随机打乱
        np.random.shuffle(available_indices)
        
        return iter(available_indices.tolist())
    
    def __len__(self):
        return len(self.dataset)


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss
    
    类似于InfoNCE，但支持多个正样本和显式的困难负样本
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        scale: float = 20.0
    ):
        super().__init__()
        self.temperature = temperature
        self.scale = scale
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            anchor_embeddings: 锚点嵌入 [batch_size, dim]
            positive_embeddings: 正样本嵌入 [batch_size, dim]
            negative_embeddings: 负样本嵌入 [batch_size, num_neg, dim] (可选)
        """
        batch_size = anchor_embeddings.shape[0]
        
        # 归一化
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # 计算与所有正样本的相似度（in-batch negatives）
        similarity_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T) * self.scale
        
        # 如果有显式负样本
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)
            # [batch_size, num_neg]
            neg_similarity = torch.bmm(
                anchor_embeddings.unsqueeze(1),
                negative_embeddings.transpose(1, 2)
            ).squeeze(1) * self.scale
            
            # 拼接
            all_similarities = torch.cat([
                similarity_matrix,
                neg_similarity
            ], dim=1)
        else:
            all_similarities = similarity_matrix
        
        # 标签是对角线位置
        labels = torch.arange(batch_size, device=anchor_embeddings.device)
        
        loss = F.cross_entropy(all_similarities, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失
    
    适用于有明确正负样本的场景
    """
    
    def __init__(self, margin: float = 0.5, distance_type: str = 'cosine'):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组损失
        
        loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
        """
        if self.distance_type == 'cosine':
            # 使用余弦距离 = 1 - 余弦相似度
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            
            pos_sim = (anchor * positive).sum(dim=1)
            neg_sim = (anchor * negative).sum(dim=1)
            
            # 我们希望 pos_sim > neg_sim + margin
            loss = F.relu(neg_sim - pos_sim + self.margin)
        else:
            # 欧氏距离
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            
            loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class GISTEmbedLoss(nn.Module):
    """
    GIST Embed Loss
    
    结合InfoNCE和困难负样本的高级损失函数
    参考: https://arxiv.org/abs/2402.16829
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        contrast_anchors: bool = True,
        contrast_positives: bool = True,
        margin: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_anchors = contrast_anchors
        self.contrast_positives = contrast_positives
        self.margin = margin
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算GIST损失
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device
        
        # 归一化
        anchor = F.normalize(anchor_embeddings, p=2, dim=1)
        positive = F.normalize(positive_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        ap_sim = torch.matmul(anchor, positive.T) / self.temperature
        
        # 可选：anchor之间的对比
        if self.contrast_anchors:
            aa_sim = torch.matmul(anchor, anchor.T) / self.temperature
            # 移除对角线
            aa_sim = aa_sim - torch.eye(batch_size, device=device) * 1e9
        
        # 可选：positive之间的对比
        if self.contrast_positives:
            pp_sim = torch.matmul(positive, positive.T) / self.temperature
            pp_sim = pp_sim - torch.eye(batch_size, device=device) * 1e9
        
        # 构建logits
        logits_list = [ap_sim]
        if self.contrast_anchors:
            logits_list.append(aa_sim)
        if self.contrast_positives:
            logits_list.append(pp_sim)
        
        # 如果有显式负样本
        if negative_embeddings is not None:
            negative = F.normalize(negative_embeddings, p=2, dim=2)
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),
                negative.transpose(1, 2)
            ).squeeze(1) / self.temperature
            
            # 应用margin过滤
            if self.margin > 0:
                pos_sim_diag = torch.diagonal(ap_sim)
                mask = neg_sim > (pos_sim_diag.unsqueeze(1) - self.margin)
                neg_sim = neg_sim * mask.float() - 1e9 * (~mask).float()
            
            logits_list.append(neg_sim)
        
        # 合并所有logits
        all_logits = torch.cat(logits_list, dim=1)
        
        # 标签
        labels = torch.arange(batch_size, device=device)
        
        loss = F.cross_entropy(all_logits, labels)
        
        return loss


@dataclass
class MultiStageTrainingConfig:
    """多阶段训练配置"""
    
    # 第一阶段：大batch size，低学习率
    stage1_batch_size: int = 64
    stage1_lr: float = 1e-5
    stage1_epochs: int = 3
    stage1_temperature: float = 0.1
    
    # 第二阶段：困难负样本，正常学习率
    stage2_batch_size: int = 32
    stage2_lr: float = 2e-5
    stage2_epochs: int = 5
    stage2_temperature: float = 0.05
    stage2_hard_negatives: int = 5
    
    # 第三阶段：微调，低学习率
    stage3_batch_size: int = 16
    stage3_lr: float = 5e-6
    stage3_epochs: int = 2
    stage3_temperature: float = 0.02


def compute_difficulty_scores(
    embedder,
    dataset: Dataset
) -> List[float]:
    """
    计算每个样本的难度分数
    
    难度定义为正样本相似度与最困难负样本相似度之差
    差值越小越难
    """
    logger.info("Computing difficulty scores...")
    
    asm_codes = [dataset[i]['asm_code'] for i in range(len(dataset))]
    source_codes = [dataset[i]['source_code'] for i in range(len(dataset))]
    
    asm_embeddings = embedder.encode_assembly(asm_codes)
    source_embeddings = embedder.encode_source(source_codes)
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(asm_embeddings, source_embeddings.T)
    
    difficulty_scores = []
    for i in range(len(dataset)):
        pos_sim = similarity_matrix[i, i]
        
        # 获取负样本相似度
        neg_sims = np.concatenate([
            similarity_matrix[i, :i],
            similarity_matrix[i, i+1:]
        ])
        
        if len(neg_sims) > 0:
            hard_neg_sim = np.max(neg_sims)
            difficulty = pos_sim - hard_neg_sim  # 正值表示容易，负值表示困难
        else:
            difficulty = 1.0  # 没有负样本
        
        difficulty_scores.append(difficulty)
    
    return difficulty_scores