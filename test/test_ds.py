#!/usr/bin/env python3
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader

# 简单的测试数据集
class DummyDataset(Dataset):
    def __init__(self, size=1000, input_dim=10, output_dim=1):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randn(size, output_dim)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def test_training():
    """测试训练功能"""
    print("=== 测试训练功能 ===")
    
    try:
        # 创建模型和数据
        model = SimpleModel()
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # DeepSpeed配置
        ds_config = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-3,
                    "warmup_num_steps": 1000
                }
            },
            "fp16": {
                "enabled": False
            },
            "steps_per_print": 10
        }
        
        # 初始化DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        # 简单训练循环
        for epoch in range(2):
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(model_engine.device)
                labels = labels.to(model_engine.device)
                
                outputs = model_engine(inputs)
                loss = nn.MSELoss()(outputs, labels)
                
                model_engine.backward(loss)
                model_engine.step()
                
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
        
        print("✓ 训练测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        return False

if __name__ == "__main__":
    test_training()