"""
Binary-to-Source Code Contrastive Learning Fine-tuning

这个脚本使用对比学习（Contrastive Learning）来微调Qwen2.5-Coder模型，
使其生成的隐藏状态表示能够更好地捕捉汇编代码和源码之间的语义相似性。

核心思想：
1. 使用InfoNCE损失函数优化表示空间
2. 正样本对：同源的汇编代码和源码
3. 负样本：batch内其他不相关的函数对
4. 使用LoRA进行参数高效微调
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, field
import argparse
from pathlib import Path
import wandb
import pandas as pd

from loss_func import HardNegativeInfoNCELossWithLabels
from playdata import BinarySourceDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 训练配置
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 对比学习配置
    temperature: float = 0.05  # InfoNCE温度参数
    embedding_dim: int = 256   # 投影后的嵌入维度
    use_projection_head: bool = True  # 是否使用投影头
    hard_negative_weight: float = 1.0  # 困难负样本权重
    
    # 数据配置
    max_seq_length: int = 512
    train_data_path: str = "data/train.json"
    val_data_path: str = "data/val.json"
    
    # 其他
    output_dir: str = "outputs"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 42
    fp16: bool = True
    use_wandb: bool = False


class ProjectionHead(nn.Module):
    """
    投影头：将隐藏状态映射到对比学习的嵌入空间
    
    使用MLP结构可以学习更好的表示，避免直接使用隐藏状态
    """
    
    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ContrastiveModel(nn.Module):
    """
    对比学习模型封装
    
    包含：
    1. 基座语言模型（Qwen2.5-Coder）
    2. 投影头（可选）
    """
    
    def __init__(
        self,
        model,
        hidden_size: int,
        projection_dim: int = 256,
        use_projection_head: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.use_projection_head = use_projection_head
        
        if use_projection_head:
            self.projection_head = ProjectionHead(
                hidden_size=hidden_size,
                projection_dim=projection_dim,
                dropout=dropout
            )
        else:
            self.projection_head = None
    
    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_strategy: str = 'mean'
    ) -> torch.Tensor:
        """
        获取序列的嵌入表示
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            pooling_strategy: 池化策略 ('last_token', 'mean', 'cls')
        
        Returns:
            embeddings: 嵌入表示
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        if pooling_strategy == 'last_token':
            # 使用最后一个非padding token的隐藏状态
            # 找到每个序列的最后一个非padding位置
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths
            ]
        elif pooling_strategy == 'mean':
            # 使用所有非padding token的平均值
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask).sum(dim=1)
            count = mask.sum(dim=1)
            embeddings = sum_embeddings / count
        elif pooling_strategy == 'eos':
            # 使用EOS token的隐藏状态（与last_token类似但更明确）
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths
            ]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        # 通过投影头
        if self.use_projection_head and self.projection_head is not None:
            embeddings = self.projection_head(embeddings)
        
        return embeddings
    
    def forward(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: torch.Tensor,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        pooling_strategy: str = 'last_token'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，获取汇编和源码的嵌入
        """
        asm_embeddings = self.get_embedding(
            asm_input_ids, asm_attention_mask, pooling_strategy
        )
        source_embeddings = self.get_embedding(
            source_input_ids, source_attention_mask, pooling_strategy
        )
        return asm_embeddings, source_embeddings


class ContrastiveTrainer:
    """
    对比学习训练器
    """
    
    def __init__(
        self,
        model: ContrastiveModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # 损失函数
        self.loss_fn = HardNegativeInfoNCELossWithLabels(
            temperature=config.temperature,
            hard_negative_weight=config.hard_negative_weight
        )
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 梯度缩放器（用于混合精度训练）
        self.scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
        
        self.global_step = 0
        self.best_mrr = 0.0
    
    def _create_optimizer(self):
        """创建优化器，对不同参数使用不同的学习率"""
        # 分离基座模型参数和投影头参数
        base_params = []
        projection_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lm_head' in name:
                    projection_params.append(param)
                else:
                    base_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.config.learning_rate},
            {'params': projection_params, 'lr': self.config.learning_rate * 10}  # 投影头使用更大学习率
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'accuracy': 0.0, 'mean_pos_sim': 0.0, 'mean_neg_sim': 0.0}
        num_batches = 0
        
        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in progress_bar:
            # 移动数据到设备
            asm_input_ids = batch['asm_input_ids'].to(self.device)
            asm_attention_mask = batch['asm_attention_mask'].to(self.device)
            source_input_ids = batch['source_input_ids'].to(self.device)
            source_attention_mask = batch['source_attention_mask'].to(self.device)
            labels = batch["label"].to(self.device)
            
            # 混合精度训练
            if self.config.fp16:
                with torch.amp.autocast('cuda'):
                    asm_emb, source_emb = self.model(
                        asm_input_ids, asm_attention_mask,
                        source_input_ids, source_attention_mask
                    )
                    loss, metrics = self.loss_fn(asm_emb, source_emb, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                asm_emb, source_emb = self.model(
                    asm_input_ids, asm_attention_mask,
                    source_input_ids, source_attention_mask
                )
                loss, metrics = self.loss_fn(asm_emb, source_emb, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            num_batches += 1
            
            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    avg_acc = total_metrics['accuracy'] / num_batches
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{avg_acc:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/accuracy': avg_acc,
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/pos_sim': total_metrics['mean_pos_sim'] / num_batches,
                            'train/neg_sim': total_metrics['mean_neg_sim'] / num_batches,
                        }, step=self.global_step)
                
                # 评估
                if self.val_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} - Eval MRR: {eval_metrics['mrr']:.4f}")
                    
                    if eval_metrics['mrr'] > self.best_mrr:
                        self.best_mrr = eval_metrics['mrr']
                        self.save_checkpoint('best')
                    
                    self.model.train()
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
        
        return {
            'loss': total_loss / num_batches,
            **{k: v / num_batches for k, v in total_metrics.items()}
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        all_asm_embeddings = []
        all_source_embeddings = []
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            asm_input_ids = batch['asm_input_ids'].to(self.device)
            asm_attention_mask = batch['asm_attention_mask'].to(self.device)
            source_input_ids = batch['source_input_ids'].to(self.device)
            source_attention_mask = batch['source_attention_mask'].to(self.device)
            
            if self.config.fp16:
                with torch.amp.autocast('cuda'):
                    asm_emb, source_emb = self.model(
                        asm_input_ids, asm_attention_mask,
                        source_input_ids, source_attention_mask
                    )
            else:
                asm_emb, source_emb = self.model(
                    asm_input_ids, asm_attention_mask,
                    source_input_ids, source_attention_mask
                )
            
            all_asm_embeddings.append(asm_emb.cpu())
            all_source_embeddings.append(source_emb.cpu())
        
        # 合并所有嵌入
        all_asm_embeddings = torch.cat(all_asm_embeddings, dim=0)
        all_source_embeddings = torch.cat(all_source_embeddings, dim=0)
        
        # 计算评估指标
        metrics = self._compute_retrieval_metrics(
            all_asm_embeddings,
            all_source_embeddings
        )
        
        if self.config.use_wandb:
            wandb.log({
                'eval/mrr': metrics['mrr'],
                'eval/recall@1': metrics['recall@1'],
                'eval/recall@5': metrics['recall@5'],
                'eval/recall@10': metrics['recall@10'],
            }, step=self.global_step)
        
        return metrics
    
    def _compute_retrieval_metrics(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """计算检索指标"""
        # L2归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        key_embeddings = F.normalize(key_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(query_embeddings, key_embeddings.T)
        
        # 获取排名
        # 对每个query，按相似度降序排列，找到正确匹配（对角线）的位置
        num_samples = query_embeddings.shape[0]
        ranks = []
        
        for i in range(num_samples):
            sims = similarity_matrix[i]
            # 按相似度降序排列
            sorted_indices = torch.argsort(sims, descending=True)
            # 找到正确答案（索引i）的位置
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # 计算指标
        mrr = np.mean(1.0 / ranks)
        recall_at_1 = np.mean(ranks <= 1)
        recall_at_5 = np.mean(ranks <= 5)
        recall_at_10 = np.mean(ranks <= 10)
        
        return {
            'mrr': mrr,
            'recall@1': recall_at_1,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
            'mean_rank': np.mean(ranks)
        }
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        self.model.model.save_pretrained(output_dir / 'lora_weights')
        
        # 保存投影头
        if self.model.use_projection_head:
            torch.save(
                self.model.projection_head.state_dict(),
                output_dir / 'projection_head.pt'
            )
        
        # 保存训练状态
        torch.save({
            'global_step': self.global_step,
            'best_mrr': self.best_mrr,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, output_dir / 'training_state.pt')
        
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def train(self):
        """完整训练流程"""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch + 1} completed. "
                f"Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
            )
            
            # 每个epoch结束时评估
            if self.val_dataloader:
                eval_metrics = self.evaluate()
                logger.info(
                    f"Eval MRR: {eval_metrics['mrr']:.4f}, "
                    f"Recall@1: {eval_metrics['recall@1']:.4f}, "
                    f"Recall@5: {eval_metrics['recall@5']:.4f}"
                )
        
        # 保存最终模型
        self.save_checkpoint('final')
        logger.info(f"Training completed. Best MRR: {self.best_mrr:.4f}")


def create_sample_data():
    """创建示例数据用于测试"""
    sample_data = [
        {
            "asm_code": """push rbp
mov rbp, rsp
mov DWORD PTR [rbp-4], edi
mov DWORD PTR [rbp-8], esi
mov eax, DWORD PTR [rbp-4]
add eax, DWORD PTR [rbp-8]
pop rbp
ret""",
            "source_code": """int add(int a, int b) {
    return a + b;
}""",
            "function_name": "add"
        },
        {
            "asm_code": """push rbp
mov rbp, rsp
mov DWORD PTR [rbp-4], edi
mov DWORD PTR [rbp-8], esi
mov eax, DWORD PTR [rbp-4]
imul eax, DWORD PTR [rbp-8]
pop rbp
ret""",
            "source_code": """int multiply(int x, int y) {
    return x * y;
}""",
            "function_name": "multiply"
        },
        {
            "asm_code": """push rbp
mov rbp, rsp
mov DWORD PTR [rbp-4], edi
cmp DWORD PTR [rbp-4], 1
jg .L2
mov eax, 1
jmp .L3
.L2:
mov eax, DWORD PTR [rbp-4]
sub eax, 1
mov edi, eax
call factorial
imul eax, DWORD PTR [rbp-4]
.L3:
pop rbp
ret""",
            "source_code": """int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}""",
            "function_name": "factorial"
        }
    ]
    
    # 复制多份以创建足够的训练数据
    expanded_data = sample_data * 20
    
    os.makedirs('data', exist_ok=True)
    
    with open('data/train.json', 'w') as f:
        json.dump(expanded_data[:50], f, indent=2)
    
    with open('data/val.json', 'w') as f:
        json.dump(expanded_data[50:], f, indent=2)
    
    logger.info("Sample data created in data/ directory")


def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning for Binary-Source Similarity')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--train_data', type=str, default='resources/datasets/train_dataset.csv')
    parser.add_argument('--val_data', type=str, default='resources/datasets/eval_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--create_sample_data', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # 创建示例数据（如果需要）
    if args.create_sample_data:
        create_sample_data()
    
    # 配置
    config = TrainingConfig(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        max_seq_length=args.max_seq_length,
        use_wandb=args.use_wandb
    )
    
    # 初始化wandb
    if config.use_wandb:
        wandb.init(
            project="binary-source-similarity",
            config=vars(config)
        )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 加载tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 对正态分布权重效果更好
        bnb_4bit_use_double_quant=True,  # 双重量化，额外节省 0.4 bit/参数
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    )
    print(f"✓ 使用 4-bit 量化 (NF4 + 双重量化)")
    
    # 加载模型
    logger.info(f"Loading model: {config.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16 if quantization_config == "none" else None,
        trust_remote_code=True,
        device_map='auto'
    )
    
    # 配置LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()
    
    # 创建对比学习模型
    hidden_size = base_model.config.hidden_size
    contrastive_model = ContrastiveModel(
        model=base_model,
        hidden_size=hidden_size,
        projection_dim=config.embedding_dim,
        use_projection_head=config.use_projection_head,
        dropout=config.lora_dropout
    )
    contrastive_model = contrastive_model.to(device)
    
    # 创建数据集和数据加载器
    logger.info("Loading datasets...")
    train_dataset = BinarySourceDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_length
    )
    
    val_dataset = None
    if os.path.exists(config.val_data_path):
        val_dataset = BinarySourceDataset(
            data_path=config.val_data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # 创建训练器并开始训练
    trainer = ContrastiveTrainer(
        model=contrastive_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=device
    )
    
    trainer.train()
    
    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()