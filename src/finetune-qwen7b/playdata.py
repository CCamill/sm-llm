import torch
from typing import List, Dict
from torch.utils.data import Dataset
import pandas as pd

class BinarySourceDataset(Dataset):
    r"""
    二进制-源码配对数据集
    
    数据格式示例:
                                                    asm_func                                           src_func  label
    0  0x5690 endbr64 | 0x5694 push    r15 | 0x5696 m...  static int updateMapping(\n  Rtree *pRtree, \n...      0
    1  0x5c20 push    r12 | 0x5c22 mov     r12, rdx |...  static int fts3ShadowName(const char *zName){\...      0
    2             0x690 endbr64 | 0x694 jmp     gzseek64  z_off_t ZEXPORT gzseek(gzFile file, z_off_t of...      1
    3  0x4790 endbr64 | 0x4794 push    r15 | 0x4796 p...  int lsmFsIntegrityCheck(lsm_db *pDb){\n  Check...      1
    4  0x840 endbr64 | 0x844 sub     rsp, 18h | 0x848...  void gh_heap_verify(gh_heap_t *heap) {\n  gh_h...      0


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
        label = item['label']
        
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
            'label': label
        }