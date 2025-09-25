"""数据分割器，用于将整合后的数据按比例分割为训练/测试/验证集"""

import os
import random
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import pandas as pd
import pickle
from datetime import datetime

from data_integrator import IntegratedSample

@dataclass
class DatasetSplit:
    """数据集分割结果"""
    train_samples: List[IntegratedSample]
    test_samples: List[IntegratedSample]
    val_samples: List[IntegratedSample]
    split_ratios: Dict[str, float]
    statistics: Dict[str, Any]

class DataSplitter:
    """数据分割器"""
    
    def __init__(self, output_dir: str = "resources/datasets/integrated_labels"):
        """
        初始化数据分割器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def split_dataset(self, 
                     samples: List[IntegratedSample],
                     train_ratio: float = 0.7,
                     test_ratio: float = 0.2,
                     val_ratio: float = 0.1,
                     shuffle: bool = True,
                     random_seed: int = 42) -> DatasetSplit:
        """
        分割数据集
        
        Args:
            samples: 样本列表
            train_ratio: 训练集比例
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            shuffle: 是否打乱顺序
            random_seed: 随机种子
            
        Returns:
            DatasetSplit: 分割结果
        """
        # 验证比例
        total_ratio = train_ratio + test_ratio + val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例总和必须为1.0，当前为{total_ratio}")
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
        
        # 打乱数据
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)
        
        # 计算分割点
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        test_size = int(total_samples * test_ratio)
        
        # 分割数据
        train_samples = samples[:train_size]
        test_samples = samples[train_size:train_size + test_size]
        val_samples = samples[train_size + test_size:]
        
        # 计算统计信息
        statistics = self._calculate_split_statistics(
            train_samples, test_samples, val_samples, total_samples
        )
        
        # 创建分割结果
        split_result = DatasetSplit(
            train_samples=train_samples,
            test_samples=test_samples,
            val_samples=val_samples,
            split_ratios={
                "train": train_ratio,
                "test": test_ratio,
                "val": val_ratio
            },
            statistics=statistics
        )
        
        self.logger.info(f"数据集分割完成:")
        self.logger.info(f"  - 训练集: {len(train_samples)} 个样本 ({len(train_samples)/total_samples:.1%})")
        self.logger.info(f"  - 测试集: {len(test_samples)} 个样本 ({len(test_samples)/total_samples:.1%})")
        self.logger.info(f"  - 验证集: {len(val_samples)} 个样本 ({len(val_samples)/total_samples:.1%})")
        
        return split_result
    
    def _calculate_split_statistics(self, 
                                  train_samples: List[IntegratedSample],
                                  test_samples: List[IntegratedSample],
                                  val_samples: List[IntegratedSample],
                                  total_samples: int) -> Dict[str, Any]:
        """
        计算分割统计信息
        
        Args:
            train_samples: 训练集样本
            test_samples: 测试集样本
            val_samples: 验证集样本
            total_samples: 总样本数
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        def _analyze_samples(samples: List[IntegratedSample], split_name: str) -> Dict[str, Any]:
            """分析单个分割的统计信息"""
            if not samples:
                return {}
            
            # 基本统计
            similar_count = sum(1 for s in samples if s.sample_type == "similar")
            dissimilar_count = sum(1 for s in samples if s.sample_type == "dissimilar")
            
            # 相似度统计
            similarities = [s.similarity_score for s in samples]
            import numpy as np
            
            return {
                "total_samples": len(samples),
                "similar_samples": similar_count,
                "dissimilar_samples": dissimilar_count,
                "similarity_stats": {
                    "mean": float(np.mean(similarities)),
                    "std": float(np.std(similarities)),
                    "min": float(np.min(similarities)),
                    "max": float(np.max(similarities)),
                    "median": float(np.median(similarities))
                }
            }
        
        return {
            "total_samples": total_samples,
            "train": _analyze_samples(train_samples, "train"),
            "test": _analyze_samples(test_samples, "test"),
            "val": _analyze_samples(val_samples, "val")
        }
    
    def save_split_to_csv(self, 
                         split_result: DatasetSplit,
                         base_filename: str = "dataset_split") -> Dict[str, str]:
        """
        保存分割结果到CSV文件
        
        Args:
            split_result: 分割结果
            base_filename: 基础文件名
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        saved_files = {}
        
        # 保存训练集
        train_file = self._save_samples_to_csv(
            split_result.train_samples, 
            f"{base_filename}_train.csv"
        )
        saved_files["train"] = train_file
        
        # 保存测试集
        test_file = self._save_samples_to_csv(
            split_result.test_samples, 
            f"{base_filename}_test.csv"
        )
        saved_files["test"] = test_file
        
        # 保存验证集
        val_file = self._save_samples_to_csv(
            split_result.val_samples, 
            f"{base_filename}_val.csv"
        )
        saved_files["val"] = val_file
        
        # 保存统计信息
        stats_file = self._save_statistics_to_csv(
            split_result.statistics,
            f"{base_filename}_statistics.csv"
        )
        saved_files["statistics"] = stats_file
        
        self.logger.info(f"分割结果已保存到CSV文件:")
        for split_name, file_path in saved_files.items():
            self.logger.info(f"  - {split_name}: {file_path}")
        
        return saved_files
    
    def _save_samples_to_csv(self, 
                            samples: List[IntegratedSample],
                            filename: str) -> str:
        """
        保存样本到CSV文件
        
        Args:
            samples: 样本列表
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备CSV数据
        csv_data = []
        for sample in samples:
            csv_data.append({
                "source_signature": sample.source_signature,
                "asm_signature": sample.asm_signature,
                "similarity_score": sample.similarity_score,
                "sample_type": sample.sample_type,
                "match_type": sample.match_type,
                "confidence": sample.confidence,
                "source_project": sample.source_func.project_name,
                "source_file": sample.source_func.file_name,
                "source_function": sample.source_func.function_name,
                "asm_project": sample.asm_func.project_name,
                "asm_file": sample.asm_func.file_name,
                "asm_function": sample.asm_func.function_name
            })
        
        # 保存为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def _save_statistics_to_csv(self, 
                               statistics: Dict[str, Any],
                               filename: str) -> str:
        """
        保存统计信息到CSV文件
        
        Args:
            statistics: 统计信息
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备统计CSV数据
        stats_data = []
        for split_name in ["train", "test", "val"]:
            if split_name in statistics:
                split_stats = statistics[split_name]
                if split_stats:
                    stats_data.append({
                        "split": split_name,
                        "total_samples": split_stats.get("total_samples", 0),
                        "similar_samples": split_stats.get("similar_samples", 0),
                        "dissimilar_samples": split_stats.get("dissimilar_samples", 0),
                        "similarity_mean": split_stats.get("similarity_stats", {}).get("mean", 0.0),
                        "similarity_std": split_stats.get("similarity_stats", {}).get("std", 0.0),
                        "similarity_min": split_stats.get("similarity_stats", {}).get("min", 0.0),
                        "similarity_max": split_stats.get("similarity_stats", {}).get("max", 0.0),
                        "similarity_median": split_stats.get("similarity_stats", {}).get("median", 0.0)
                    })
        
        # 保存为CSV
        df = pd.DataFrame(stats_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    def save_split_to_pkl(self, 
                         split_result: DatasetSplit,
                         filename: str = "dataset_split.pkl") -> str:
        """
        保存分割结果到PKL文件
        
        Args:
            split_result: 分割结果
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备保存的数据
        save_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "split_ratios": split_result.split_ratios,
                "statistics": split_result.statistics
            },
            "train_samples": split_result.train_samples,
            "test_samples": split_result.test_samples,
            "val_samples": split_result.val_samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"分割结果已保存到PKL文件: {filepath}")
        return filepath
    
    def load_split_from_pkl(self, filename: str) -> DatasetSplit:
        """
        从PKL文件加载分割结果
        
        Args:
            filename: 文件名
            
        Returns:
            DatasetSplit: 分割结果
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        split_result = DatasetSplit(
            train_samples=data["train_samples"],
            test_samples=data["test_samples"],
            val_samples=data["val_samples"],
            split_ratios=data["metadata"]["split_ratios"],
            statistics=data["metadata"]["statistics"]
        )
        
        self.logger.info(f"从PKL文件加载了分割结果: {filepath}")
        return split_result
    
    def validate_split(self, split_result: DatasetSplit) -> bool:
        """
        验证分割结果
        
        Args:
            split_result: 分割结果
            
        Returns:
            bool: 验证是否通过
        """
        # 检查样本数量
        total_samples = (len(split_result.train_samples) + 
                        len(split_result.test_samples) + 
                        len(split_result.val_samples))
        
        if total_samples == 0:
            self.logger.error("分割结果中没有样本")
            return False
        
        # 检查比例
        train_ratio = len(split_result.train_samples) / total_samples
        test_ratio = len(split_result.test_samples) / total_samples
        val_ratio = len(split_result.val_samples) / total_samples
        
        expected_train = split_result.split_ratios["train"]
        expected_test = split_result.split_ratios["test"]
        expected_val = split_result.split_ratios["val"]
        
        tolerance = 0.05  # 5%的容差
        
        if (abs(train_ratio - expected_train) > tolerance or
            abs(test_ratio - expected_test) > tolerance or
            abs(val_ratio - expected_val) > tolerance):
            self.logger.warning(f"分割比例与预期不符:")
            self.logger.warning(f"  预期: train={expected_train:.1%}, test={expected_test:.1%}, val={expected_val:.1%}")
            self.logger.warning(f"  实际: train={train_ratio:.1%}, test={test_ratio:.1%}, val={val_ratio:.1%}")
            return False
        
        self.logger.info("分割结果验证通过")
        return True
