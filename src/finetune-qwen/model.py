import torch 
import torch.nn as nn
from typing import Tuple

class ProjectionHead(nn.Module):
    """
    投影头：将隐藏状态映射到对比学习的嵌入空间
    
    使用MLP结构可以学习更好的表示，避免直接使用隐藏状态
    """
    
    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        dropout: float = 0.2
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
            embeddings = embeddings.to(self.projection_head.projection[0].weight.dtype)
            embeddings = self.projection_head(embeddings)
        
        return embeddings
    
    def forward(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: torch.Tensor,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        pooling_strategy: str = 'mean'
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