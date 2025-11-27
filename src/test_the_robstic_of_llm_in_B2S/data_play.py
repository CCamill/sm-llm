import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class FuncDataset(Dataset):
    def __init__(self, csv_file: str, items_num: int = None, ratio_pos_neg: int = 0.5, task_type: str = "matching"):
        """
        自定义数据集类，用于加载函数对数据。
        Args:
            csv_file: 包含函数对的CSV文件路径
            items_num: 加载的数据项数量, None表示加载全部
            ratio_pos_neg: 正负样本比例 [0.5, 1.0]
        """
        super().__init__()
        data_frame = pd.read_csv(csv_file)
        self.data_frame = data_frame if items_num is None else data_frame.iloc[:items_num]
        self.ratio_pos_neg = ratio_pos_neg
        self.task_type = task_type
    
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
            return asm_func_1, src_func_1
    
