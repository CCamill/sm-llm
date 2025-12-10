"""
支持显式标签的对比学习损失函数

适用于数据集返回格式:
{
    'asm_input_ids': ...,
    'asm_attention_mask': ...,
    'source_input_ids': ...,
    'source_attention_mask': ...,
    'label': 0 或 1  # 0表示负样本，1表示正样本
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class LabeledContrastiveLoss(nn.Module):
    """
    基于显式标签的对比学习损失
    
    支持batch中同时包含正样本对（label=1）和负样本对（label=0）
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        hard_negative_weight: float = 1.0,
        margin: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比学习损失
        
        Args:
            asm_embeddings: 汇编代码嵌入 [batch_size, embedding_dim]
            source_embeddings: 源代码嵌入 [batch_size, embedding_dim]
            labels: 标签 [batch_size]，1表示正样本对，0表示负样本对
        
        Returns:
            loss: 损失值
            metrics: 评估指标
        """
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device
        
        # L2归一化
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        
        # 计算每对样本的相似度（点积，因为已经归一化所以等于余弦相似度）
        # [batch_size]
        pairwise_sim = (asm_embeddings * source_embeddings).sum(dim=1) / self.temperature
        
        # 分离正负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        # 获取正负样本的相似度
        pos_sim = pairwise_sim[pos_mask] if num_pos > 0 else None
        neg_sim = pairwise_sim[neg_mask] if num_neg > 0 else None
        
        # ============ 计算损失 ============
        
        total_loss = torch.tensor(0.0, device=device)
        
        # 方法1: 对比损失 (Contrastive Loss)
        # 正样本：希望相似度高 -> 距离小
        # 负样本：希望相似度低 -> 距离大于margin
        
        if num_pos > 0:
            # 正样本损失：最大化相似度（最小化 1 - sim）
            pos_loss = (1 - pos_sim * self.temperature).mean()
            total_loss = total_loss + pos_loss
        
        if num_neg > 0:
            # 负样本损失：确保相似度低于阈值
            # 使用hinge loss: max(0, sim - margin)
            neg_loss = F.relu(neg_sim * self.temperature + self.margin).mean()
            total_loss = total_loss + self.hard_negative_weight * neg_loss
        
        # ============ 计算指标 ============
        
        with torch.no_grad():
            metrics = {}
            
            if num_pos > 0:
                metrics['mean_pos_sim'] = (pos_sim * self.temperature).mean().item()
                metrics['num_pos'] = num_pos
            else:
                metrics['mean_pos_sim'] = 0.0
                metrics['num_pos'] = 0
            
            if num_neg > 0:
                metrics['mean_neg_sim'] = (neg_sim * self.temperature).mean().item()
                metrics['num_neg'] = num_neg
                # 困难负样本（相似度最高的负样本）
                metrics['hard_neg_sim'] = (neg_sim.max() * self.temperature).item()
            else:
                metrics['mean_neg_sim'] = 0.0
                metrics['num_neg'] = 0
                metrics['hard_neg_sim'] = 0.0
            
            # 计算准确率（使用0.5作为阈值）
            threshold = 0.5 / self.temperature
            predictions = (pairwise_sim > threshold).float()
            metrics['accuracy'] = (predictions == labels.float()).float().mean().item()
            
            # 正负样本分离度
            if num_pos > 0 and num_neg > 0:
                metrics['sim_gap'] = metrics['mean_pos_sim'] - metrics['mean_neg_sim']
            else:
                metrics['sim_gap'] = 0.0
        
        return total_loss, metrics


    """
    三元组损失：专门为混合正负样本的数据设计
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, asm_embeddings, source_embeddings, labels):
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device
        
        # L2归一化
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        
        # 分离正负样本
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            metrics = {'loss': 0.0, 'num_pos': len(pos_indices), 'num_neg': len(neg_indices)}
            return torch.tensor(0.0, device=device), metrics
        
        total_loss = 0.0
        num_triplets = 0
        
        # 为每个正样本对构建三元组
        for pos_idx in pos_indices:
            anchor = asm_embeddings[pos_idx]      # 汇编代码
            positive = source_embeddings[pos_idx] # 对应的源代码（正样本）
            
            # 随机选择一个负样本
            if len(neg_indices) > 0:
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                negative = source_embeddings[neg_idx]
                
                loss = self.triplet_loss(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                total_loss += loss
                num_triplets += 1
        
        if num_triplets > 0:
            total_loss = total_loss / num_triplets
        
        # 计算指标
        with torch.no_grad():
            pairwise_sim = (asm_embeddings * source_embeddings).sum(dim=1)
            predictions = (pairwise_sim > 0).float()
            accuracy = (predictions == labels.float()).float().mean().item()
            
            pos_sim = pairwise_sim[pos_mask].mean().item() if len(pos_mask) > 0 else 0.0
            neg_sim = pairwise_sim[neg_mask].mean().item() if len(neg_mask) > 0 else 0.0
            
            metrics = {
                'loss': total_loss.item() if num_triplets > 0 else 0.0,
                'accuracy': accuracy,
                'num_pos': len(pos_indices),
                'num_neg': len(neg_indices),
                'mean_pos_sim': pos_sim,
                'mean_neg_sim': neg_sim,
                'sim_gap': pos_sim - neg_sim
            }
        
        return total_loss, metrics

class PairwiseCosineLoss(nn.Module):
    """
    适用于二分类验证任务：
    Label=1: 拉近 (maximize cosine similarity)
    Label=0: 推远 (minimize cosine similarity, with margin)
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        # PyTorch 自带的 CosineEmbeddingLoss
        # target=1 表示相似，target=-1 表示不相似
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, asm_embeddings, source_embeddings, labels):
        # 转换标签：0/1 -> -1/1
        # 假设输入 labels 是 0 和 1
        targets = labels.clone().float()
        targets[targets == 0] = -1  # 将 0 变为 -1
        
        loss = self.loss_fn(asm_embeddings, source_embeddings, targets)
        
        # 计算简单的二分类准确率（基于阈值）
        with torch.no_grad():
            cos_sim = F.cosine_similarity(asm_embeddings, source_embeddings)
            # 简单的阈值判定，这里假设 > margin 是正样本（具体阈值需根据业务调整）
            preds = (cos_sim > self.margin).long()
            acc = (preds == labels).float().mean()
            
            metrics = {
                'loss': loss.item(),
                'accuracy': acc.item(),
                'mean_pos_sim': cos_sim[labels==1].mean().item() if (labels==1).any() else 0,
                'mean_neg_sim': cos_sim[labels==0].mean().item() if (labels==0).any() else 0
            }
            
        return loss, metrics

class VectorizedLabeledInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, asm_embeddings, source_embeddings, labels):
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device

        # 1. L2 归一化
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)

        # 2. 计算相似度矩阵 [batch_size, batch_size]
        # sim_matrix[i][j] 表示 asm[i] 和 source[j] 的相似度
        sim_matrix = torch.matmul(asm_embeddings, source_embeddings.T) / self.temperature

        # 3. 构造 InfoNCE 的 Logits 和 Labels
        # 对于 InfoNCE，asm[i] 的目标（正样本）应该是 source[i]
        # 所以对于第 i 行，正确的类别索引就是 i
        target_indices = torch.arange(batch_size, device=device)

        # 4. 只选取 label=1 的样本计算 Loss
        # 即使是 label=0 的样本，它们仍然作为 label=1 样本的分母（负样本）参与了计算，没有完全浪费
        pos_mask = (labels == 1)
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device), {}

        # 5. 计算 CrossEntropy
        # 我们只提取 pos_mask 为 True 的行来计算 Loss
        loss = F.cross_entropy(sim_matrix[pos_mask], target_indices[pos_mask])

        # --- 计算简单指标 ---
        with torch.no_grad():
            # 这里的准确率是指：在Batch内，每个asm能否正确检索到对应的source
            preds = torch.argmax(sim_matrix, dim=1)
            correct = (preds == target_indices) & pos_mask # 只看正样本对是否检索正确
            acc = correct.sum() / pos_mask.sum()
            
            metrics = {
                'loss': loss.item(),
                'accuracy': acc.item(),
                'num_pos_pairs': pos_mask.sum().item()
            }

        return loss, metrics

class HardNegativeInfoNCELossWithLabels(nn.Module):
    """
    带显式标签的困难负样本InfoNCE损失
    
    核心思想：
    1. 对每个正样本对，使用InfoNCE让正样本相似度最高
    2. 显式负样本（label=0）获得更高的权重，产生更大的惩罚
    3. 额外的margin损失确保正负样本有足够间隔
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        hard_negative_weight: float = 1.0,
        margin: float = 0.2,
        explicit_neg_weight: float = 2.0  # 显式负样本的权重倍数
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
        self.explicit_neg_weight = explicit_neg_weight
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            asm_embeddings: 汇编代码嵌入 [batch_size, embedding_dim]
            source_embeddings: 源代码嵌入 [batch_size, embedding_dim]
            labels: 标签 [batch_size]，1表示正样本对，0表示负样本对
        
        Returns:
            total_loss: 总损失
            metrics: 评估指标
        """
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device
        
        # L2归一化，使得点积等于余弦相似度
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        # sim_matrix[i,j] = cosine_sim(asm_i, source_j) / temperature
        sim_matrix = torch.matmul(asm_embeddings, source_embeddings.T) / self.temperature
        
        # 分离正负样本
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]
        
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        
        # 边界情况：没有正样本
        if num_pos == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'accuracy': 0.0, 'mean_pos_sim': 0.0, 'mean_neg_sim': 0.0,
                'num_pos': 0, 'num_neg': num_neg, 'sim_gap': 0.0
            }
        
        # ============ 计算加权InfoNCE损失 ============
        
        total_loss = torch.tensor(0.0, device=device)
        pos_sims = []
        all_neg_sims = []
        
        for idx in pos_indices:
            idx_val = idx.item()
            
            # ------ 正样本相似度 ------
            # asm_i 与其配对的 source_i（对角线元素）
            pos_sim = sim_matrix[idx_val, idx_val]
            pos_sims.append(pos_sim)
            
            # ------ 负样本相似度（in-batch negatives）------
            # asm_i 与所有其他 source_j (j != i)
            all_indices = torch.arange(batch_size, device=device)
            in_batch_neg_mask = (all_indices != idx_val)
            in_batch_neg_sims = sim_matrix[idx_val, in_batch_neg_mask]  # [batch_size - 1]
            
            all_neg_sims.append(in_batch_neg_sims)
            
            # ------ 计算权重 ------
            # 显式负样本获得更高权重
            weights = torch.ones(batch_size - 1, device=device)
            
            # 获取负样本对应的原始索引
            neg_original_indices = all_indices[in_batch_neg_mask]
            
            for j in range(len(neg_original_indices)):
                orig_idx = neg_original_indices[j].item()
                if labels[orig_idx] == 0:  # 显式负样本
                    weights[j] = self.explicit_neg_weight
            
            # ------ 方法1：加权交叉熵（推荐）------
            # 构建logits: [正样本, 负样本1, 负样本2, ...]
            logits = torch.cat([pos_sim.unsqueeze(0), in_batch_neg_sims])  # [batch_size]
            
            # 构建权重: [1.0, w1, w2, ...]
            # 正样本权重为1，负样本权重根据是否为显式负样本决定
            full_weights = torch.cat([torch.ones(1, device=device), weights])
            
            # 加权softmax的实现：
            # P_weighted(i) = w_i * exp(s_i) / sum_j(w_j * exp(s_j))
            # log P_weighted(0) = s_0 + log(w_0) - log(sum_j(w_j * exp(s_j)))
            #                   = s_0 - logsumexp(s + log(w))  (因为log(w_0)=0)
            
            log_weights = full_weights.log()
            weighted_logsumexp = torch.logsumexp(logits + log_weights, dim=0)
            log_prob_pos = logits[0] - weighted_logsumexp  # 正样本的对数概率
            
            loss_i = -log_prob_pos  # 负对数似然
            total_loss = total_loss + loss_i
        
        # 对正样本数量取平均
        total_loss = total_loss / num_pos
        
        # ============ 困难负样本惩罚（Margin Loss）============
        
        pos_sims_tensor = torch.stack(pos_sims)
        
        if num_neg > 0 and self.hard_negative_weight > 0:
            # 显式负样本的配对相似度（对角线上的负样本对）
            explicit_neg_sims = sim_matrix[neg_indices, neg_indices]  # [num_neg]
            hard_neg_sim = explicit_neg_sims.max()
            
            # Margin损失：确保 avg(正样本相似度) > max(负样本相似度) + margin
            # 即：max(负样本) - avg(正样本) + margin < 0
            # 损失：max(0, max(负样本) - avg(正样本) + margin)
            hard_neg_penalty = F.relu(hard_neg_sim - pos_sims_tensor.mean() + self.margin)
            total_loss = total_loss + self.hard_negative_weight * hard_neg_penalty
        
        # ============ 计算评估指标 ============
        
        with torch.no_grad():
            metrics = {
                'num_pos': num_pos,
                'num_neg': num_neg,
                'mean_pos_sim': (pos_sims_tensor.mean() * self.temperature).item(),
            }
            
            if len(all_neg_sims) > 0:
                all_neg_tensor = torch.cat(all_neg_sims) * self.temperature
                metrics['mean_neg_sim'] = all_neg_tensor.mean().item()
                metrics['hard_neg_sim'] = all_neg_tensor.max().item()
            else:
                metrics['mean_neg_sim'] = 0.0
                metrics['hard_neg_sim'] = 0.0
            
            metrics['sim_gap'] = metrics['mean_pos_sim'] - metrics['mean_neg_sim']
            
            # 配对准确率
            pairwise_sim = torch.diagonal(sim_matrix) * self.temperature
            threshold = 0.5
            predictions = (pairwise_sim > threshold).float()
            metrics['accuracy'] = (predictions == labels.float()).float().mean().item()
            
            # 检索准确率：对于正样本，检查是否能正确找到配对的source
            if num_pos > 0:
                retrieval_correct = 0
                for idx in pos_indices:
                    predicted_source = sim_matrix[idx].argmax().item()
                    if predicted_source == idx.item():
                        retrieval_correct += 1
                metrics['retrieval_acc'] = retrieval_correct / num_pos
            else:
                metrics['retrieval_acc'] = 0.0
        
        return total_loss, metrics

class InfoNCELoss(nn.Module):
    """
    InfoNCE对比学习损失函数
    
    InfoNCE = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    其中：
    - z_i, z_j 是正样本对（同源的汇编和源码）
    - z_k 是所有样本（包括正负样本）
    - τ 是温度参数
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        hard_negative_weight: float = 1.0
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            asm_embeddings: 汇编代码嵌入 [batch_size, embedding_dim]
            source_embeddings: 源代码嵌入 [batch_size, embedding_dim]
            hard_negative_weight: 困难负样本权重
        
        Returns:
            loss: 对比学习损失
        """
        batch_size = asm_embeddings.shape[0]
        
        # L2归一化
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        # asm_embeddings @ source_embeddings.T -> [batch_size, batch_size]
        # 对角线元素是正样本对的相似度
        similarity_matrix = torch.matmul(asm_embeddings, source_embeddings.T) / self.temperature
        
        # 标签：对角线位置是正样本
        labels = torch.arange(batch_size, device=asm_embeddings.device)
        
        # 计算损失（双向：asm->source 和 source->asm）
        loss_asm_to_source = F.cross_entropy(similarity_matrix, labels)
        loss_source_to_asm = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_asm_to_source + loss_source_to_asm) / 2
        
        return loss


class HardNegativeInfoNCELoss(nn.Module):
    """
    带困难负样本挖掘的InfoNCE损失
    
    困难负样本是那些与查询样本相似但实际上是负样本的例子，
    它们对模型学习更有帮助。
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        hard_negative_weight: float = 1.0,
        margin: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算带困难负样本的InfoNCE损失
        """
        batch_size = asm_embeddings.shape[0]
        
        # L2归一化
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(asm_embeddings, source_embeddings.T) / temperature
        
        # 正样本掩码（对角线）
        pos_mask = torch.eye(batch_size, device=asm_embeddings.device).bool()
        
        # 获取正样本相似度
        pos_sim = sim_matrix[pos_mask]  # [batch_size]
        
        # 负样本相似度（非对角线元素）
        neg_mask = ~pos_mask
        neg_sim = sim_matrix[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]
        
        # 困难负样本：选择最相似的负样本
        hard_neg_sim, _ = neg_sim.max(dim=1)  # [batch_size]
        
        # 计算损失
        # 基础InfoNCE损失
        labels = torch.arange(batch_size, device=asm_embeddings.device)
        
        # 计算双向
        # 正向
        loss_a2s = F.cross_entropy(sim_matrix, labels)
        # 反向
        loss_s2a = F.cross_entropy(sim_matrix.T, labels)
        loss_basic = (loss_a2s + loss_s2a) / 2
        
        # 困难负样本惩罚：确保正样本比困难负样本高出margin
        hard_neg_penalty = F.relu(hard_neg_sim - pos_sim + self.margin / temperature).mean()
        
        total_loss = loss_basic + self.hard_negative_weight * hard_neg_penalty
        
        # 计算一些指标
        with torch.no_grad():
            accuracy = (sim_matrix.argmax(dim=1) == labels).float().mean().item()
            mean_pos_sim = pos_sim.mean().item() * temperature
            mean_neg_sim = neg_sim.mean().item() * temperature
        
        metrics = {
            'accuracy': accuracy,
            'mean_pos_sim': mean_pos_sim,
            'mean_neg_sim': mean_neg_sim,
            'hard_neg_penalty': hard_neg_penalty.item()
        }
        
        return total_loss, metrics

# ============ 简化版本：最推荐使用 ============

class SimpleContrastiveLossWithLabels(nn.Module):
    """
    简化版对比损失（推荐）
    
    结合了：
    1. 配对损失：直接优化正负样本对的相似度
    2. InfoNCE：利用batch内其他样本作为额外负样本
    3. Margin损失：确保正负样本有足够的间隔
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.3,
        lambda_pair: float = 1.0,      # 配对损失权重
        lambda_infonce: float = 1.0,   # InfoNCE损失权重
        lambda_margin: float = 0.5     # Margin损失权重
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.lambda_pair = lambda_pair
        self.lambda_infonce = lambda_infonce
        self.lambda_margin = lambda_margin
    
    def forward(
        self,
        asm_embeddings: torch.Tensor,
        source_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        """
        batch_size = asm_embeddings.shape[0]
        device = asm_embeddings.device
        
        # L2归一化
        asm_emb = F.normalize(asm_embeddings, p=2, dim=1)
        src_emb = F.normalize(source_embeddings, p=2, dim=1)
        
        # 配对相似度 [batch_size]
        pairwise_sim = (asm_emb * src_emb).sum(dim=1)
        
        # 完整相似度矩阵 [batch_size, batch_size]
        sim_matrix = torch.matmul(asm_emb, src_emb.T)
        
        # 分离正负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        total_loss = torch.tensor(0.0, device=device)
        
        # ============ 1. 配对损失 ============
        if self.lambda_pair > 0:
            # 正样本：最大化相似度 -> 最小化 (1 - sim)
            # 负样本：最小化相似度 -> 最小化 max(0, sim + margin)
            
            if num_pos > 0:
                pos_pair_loss = (1 - pairwise_sim[pos_mask]).mean()
            else:
                pos_pair_loss = torch.tensor(0.0, device=device)
            
            if num_neg > 0:
                neg_pair_loss = F.relu(pairwise_sim[neg_mask] + self.margin).mean()
            else:
                neg_pair_loss = torch.tensor(0.0, device=device)
            
            pair_loss = pos_pair_loss + neg_pair_loss
            total_loss = total_loss + self.lambda_pair * pair_loss
        
        # ============ 2. InfoNCE损失（只对正样本） ============
        if self.lambda_infonce > 0 and num_pos > 0:
            pos_indices = torch.where(pos_mask)[0]
            
            infonce_loss = torch.tensor(0.0, device=device)
            for idx in pos_indices:
                # 正样本在对角线上
                logits = sim_matrix[idx] / self.temperature
                target = idx
                infonce_loss = infonce_loss + F.cross_entropy(
                    logits.unsqueeze(0), 
                    target.unsqueeze(0)
                )
            
            infonce_loss = infonce_loss / num_pos
            total_loss = total_loss + self.lambda_infonce * infonce_loss
        
        # ============ 3. Margin损失 ============
        if self.lambda_margin > 0 and num_pos > 0 and num_neg > 0:
            pos_sim = pairwise_sim[pos_mask]
            neg_sim = pairwise_sim[neg_mask]
            
            # 所有正负样本对之间的margin
            # 每个正样本应该比每个负样本高出margin
            # [num_pos, 1] - [1, num_neg] -> [num_pos, num_neg]
            margin_violations = F.relu(
                neg_sim.unsqueeze(0) - pos_sim.unsqueeze(1) + self.margin
            )
            margin_loss = margin_violations.mean()
            total_loss = total_loss + self.lambda_margin * margin_loss
        
        # ============ 计算指标 ============
        with torch.no_grad():
            metrics = {
                'num_pos': num_pos,
                'num_neg': num_neg,
            }
            
            if num_pos > 0:
                metrics['mean_pos_sim'] = pairwise_sim[pos_mask].mean().item()
            else:
                metrics['mean_pos_sim'] = 0.0
            
            if num_neg > 0:
                metrics['mean_neg_sim'] = pairwise_sim[neg_mask].mean().item()
                metrics['hard_neg_sim'] = pairwise_sim[neg_mask].max().item()
            else:
                metrics['mean_neg_sim'] = 0.0
                metrics['hard_neg_sim'] = 0.0
            
            metrics['sim_gap'] = metrics['mean_pos_sim'] - metrics['mean_neg_sim']
            
            # 使用动态阈值计算准确率
            if num_pos > 0 and num_neg > 0:
                threshold = (metrics['mean_pos_sim'] + metrics['mean_neg_sim']) / 2
            else:
                threshold = 0.5
            
            predictions = (pairwise_sim > threshold).float()
            metrics['accuracy'] = (predictions == labels.float()).float().mean().item()
            
            # 如果有足够样本，计算AUC相关指标
            if num_pos > 0 and num_neg > 0:
                # 计算有多少正样本的相似度高于负样本
                correct_pairs = (pairwise_sim[pos_mask].unsqueeze(1) > 
                               pairwise_sim[neg_mask].unsqueeze(0)).float()
                metrics['pairwise_accuracy'] = correct_pairs.mean().item()
        
        return total_loss, metrics