import torch
from typing import List, Dict
from torch.utils.data import Dataset
import pandas as pd
import random

class BinarySourceDataset(Dataset):
    r"""
    二进制-源码配对数据集
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        asm_prompt_template: str = None,
        source_prompt_template: str = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.asm_prompt_template = "Analyze the following assembly code and understand its semantic meaning:\n"\
            "```asm\n{code}\n```\n"\
            "Semantic representation:"
        self.source_prompt_template = "Analyze the following source code and understand its semantic meaning:\n"\
            "```c\n{code}\n```\n"\
            "Semantic representation:"
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = pd.read_csv(data_path)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data.iloc[idx]
        
        # 构建输入文本
        asm_text = self.asm_prompt_template.format(code=item['asm_func'])
        source_text = self.source_prompt_template.format(code=item['src_func'])
        
        # Tokenize
        asm_encoding = self.tokenizer(
            asm_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'asm_input_ids': asm_encoding['input_ids'].squeeze(0),
            'asm_attention_mask': asm_encoding['attention_mask'].squeeze(0),
            'source_input_ids': source_encoding['input_ids'].squeeze(0),
            'source_attention_mask': source_encoding['attention_mask'].squeeze(0),
        }
    

class FuncDataset(Dataset):
    def __init__(self, csv_file: str, items_num: int = None, ratio_pos_neg: int = 0.5, task_type: str = "matching", asm_prompt_template:str=None, source_prompt_template:str=None):
        """
        自定义数据集类，用于加载函数对数据。
        Args:
            csv_file: 包含函数对的CSV文件路径
            items_num: 加载的数据项数量, None表示加载全部
            ratio_pos_neg: 正负样本比例 [0.5, 1.0]
        """
        super().__init__()
        df = pd.read_csv(csv_file)
        data_frame = df[df['label'] != 0]
        self.data_frame = data_frame if items_num is None else data_frame.iloc[:items_num]
        self.ratio_pos_neg = ratio_pos_neg
        self.task_type = task_type
        self.asm_prompt_template = asm_prompt_template or (
            "Analyze the following assembly code and understand its semantic meaning:\n"
            "```asm\n{code}\n```\n"
            "Semantic representation:"
        )
        self.source_prompt_template = source_prompt_template or (
            "Analyze the following source code and understand its semantic meaning:\n"
            "```c\n{code}\n```\n"
            "Semantic representation:"
        )
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        src_func_1 = row['src_func']
        asm_func_1 = row['asm_func']
        if self.task_type=="matching":
            ftype=random.randint(0,len(self.data_frame)-1)
            while ftype==idx:
                ftype=random.randint(0,len(self.data_frame)-1)
            asm_func_2 = self.data_frame.iloc[ftype]['asm_func']
            src_func_2 = self.data_frame.iloc[ftype]['src_func']
            pair1 = (asm_func_1, src_func_1, 1)  # Positive pair 
            pair2 = (asm_func_2, src_func_1, 0)  # Negative pair 
            pair3 = (asm_func_1, src_func_2, 0)  # Negative pair
            if int(self.ratio_pos_neg) == 1.0:
                return pair1, pair2
            else:
                return pair1, pair2, pair3
        if self.task_type=="selection":
            asm_func_1 = self.asm_prompt_template.format(code=asm_func_1)
            src_func_1 = self.source_prompt_template.format(code=src_func_1)
            return asm_func_1, src_func_1, 1