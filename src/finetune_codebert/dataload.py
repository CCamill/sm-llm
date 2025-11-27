import json
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import mmap
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class FunctionSliceDataset(Dataset):
    def __init__(self, root_dir, max_cached_files=100, shuffle=True):
        """
        Args:
            root_dir: 数据集根目录（包含多个项目文件夹）
            max_cached_files: 内存中最大缓存的json文件数
            shuffle: 是否打乱数据顺序
        """
        self.root_dir = Path(root_dir)
        self.max_cached_files = max_cached_files
        self.shuffle = shuffle
        
        # 1. 建立文件索引
        self.json_files = list(self.root_dir.glob('*/*.json'))
        self.file_indices = self._build_file_indices()
        
        # 2. 初始化缓存系统
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 3. 预加载元数据
        self.total_slices = sum(len(v) for v in self.file_indices.values())
        self._prefetch_next_batch_files()

    def _build_file_indices(self):
        """构建文件到函数切片的映射索引"""
        indices = {}
        for json_file in self.json_files:
            with open(json_file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    data = json.loads(mm.read().decode('utf-8'))
                    indices[str(json_file)] = [
                        (i, len(data['functions']))  # (文件内切片起始索引, 切片数量)
                        for i in range(len(data['functions']))
                    ]
        return indices

    @lru_cache(maxsize=100)
    def _load_json_file(self, file_path):
        """带缓存的json文件加载"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _prefetch_next_batch_files(self):
        """异步预读取下一批可能用到的文件"""
        if len(self.cache) < self.max_cached_files:
            sample_files = random.sample(self.json_files, 
                                       min(10, len(self.json_files)))
            for f in sample_files:
                if str(f) not in self.cache:
                    self.executor.submit(self._load_json_file, str(f))

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        # 1. 定位到具体文件及切片
        file_path, slice_idx = self._locate_slice(idx)
        
        # 2. 获取文件数据（优先从缓存读取）
        if file_path not in self.cache:
            self.cache[file_path] = self._load_json_file(file_path)
            self._prefetch_next_batch_files()  # 触发下一批预读取
            
        # 3. 提取特定函数切片
        return self._process_slice(self.cache[file_path]['functions'][slice_idx])

    def _locate_slice(self, global_idx):
        """将全局索引转换为(文件路径, 文件内切片索引)"""
        for file_path, slices in self.file_indices.items():
            if global_idx < len(slices):
                return file_path, slices[global_idx][0]
            global_idx -= len(slices)
        raise IndexError("Index out of range")

    def _process_slice(self, slice_data):
        """处理单个函数切片（可根据需求自定义）"""
        return {
            'code': slice_data['code'],
            'metadata': slice_data.get('meta', {})
        }
    
