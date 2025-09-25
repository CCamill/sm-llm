"""数据整合器，用于合并相似和不相似样本"""

import os
import json
import pickle
import random
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from data_loader import FunctionData
from dissimilar_data_loader import DissimilarPair, FunctionSignature

@dataclass
class IntegratedSample:
    """整合后的样本"""
    source_func: FunctionData
    asm_func: FunctionData
    similarity_score: float
    sample_type: str  # "similar" or "dissimilar"
    match_type: str  # "exact", "fuzzy", "llm_generated", "dissimilar"
    confidence: float
    source_signature: str
    asm_signature: str
    metadata: Dict[str, Any]

class DataIntegrator:
    """数据整合器"""
    
    def __init__(self, output_dir: str = "resources/datasets/integrated_labels"):
        """
        初始化数据整合器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def load_similar_samples(self, similar_dir: str) -> List[IntegratedSample]:
        """
        加载相似样本
        
        Args:
            similar_dir: 相似样本目录
            
        Returns:
            List[IntegratedSample]: 相似样本列表
        """
        samples = []
        
        if not os.path.exists(similar_dir):
            self.logger.warning(f"相似样本目录不存在: {similar_dir}")
            return samples
        
        # 遍历所有JSON文件
        for filename in os.listdir(similar_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(similar_dir, filename)
                file_samples = self._load_similar_samples_from_file(file_path)
                samples.extend(file_samples)
        
        self.logger.info(f"加载了 {len(samples)} 个相似样本")
        return samples
    
    def _load_similar_samples_from_file(self, file_path: str) -> List[IntegratedSample]:
        """
        从JSON文件加载相似样本
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[IntegratedSample]: 相似样本列表
        """
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for sample_data in data.get('samples', []):
                # 创建FunctionData对象
                source_func = FunctionData(
                    function_id=sample_data['source_function']['function_id'],
                    function_name=sample_data['source_function']['function_name'],
                    signature=sample_data['source_function']['signature'],
                    body=sample_data['source_function']['body'],
                    full_definition=sample_data['source_function']['full_definition'],
                    source_file=sample_data['source_function']['source_file'],
                    project_name=sample_data['source_function']['file_name'].split('/')[0] if '/' in sample_data['source_function']['file_name'] else 'unknown',
                    file_name=sample_data['source_function']['file_name']
                )
                
                asm_func = FunctionData(
                    function_id=sample_data['asm_function']['function_id'],
                    function_name=sample_data['asm_function']['function_name'],
                    signature=sample_data['asm_function']['signature'],
                    body=sample_data['asm_function']['body'],
                    full_definition=sample_data['asm_function']['full_definition'],
                    source_file=sample_data['asm_function']['source_file'],
                    project_name=sample_data['asm_function']['file_name'].split('/')[0] if '/' in sample_data['asm_function']['file_name'] else 'unknown',
                    file_name=sample_data['asm_function']['file_name']
                )
                
                # 创建签名
                source_signature = f"{source_func.project_name}+{source_func.file_name}+{source_func.function_name}"
                asm_signature = f"{asm_func.project_name}+{asm_func.file_name}+{asm_func.function_name}"
                
                # 创建整合样本
                sample = IntegratedSample(
                    source_func=source_func,
                    asm_func=asm_func,
                    similarity_score=sample_data['similarity_label'],
                    sample_type="similar",
                    match_type=sample_data['match_type'],
                    confidence=sample_data['confidence'],
                    source_signature=source_signature,
                    asm_signature=asm_signature,
                    metadata=sample_data['metadata']
                )
                
                samples.append(sample)
                
        except Exception as e:
            self.logger.error(f"加载相似样本文件失败 {file_path}: {e}")
        
        return samples
    
    def load_dissimilar_samples(self, dissimilar_dir: str) -> List[IntegratedSample]:
        """
        加载不相似样本
        
        Args:
            dissimilar_dir: 不相似样本目录
            
        Returns:
            List[IntegratedSample]: 不相似样本列表
        """
        samples = []
        
        if not os.path.exists(dissimilar_dir):
            self.logger.warning(f"不相似样本目录不存在: {dissimilar_dir}")
            return samples
        
        # 遍历所有PKL文件
        for filename in os.listdir(dissimilar_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(dissimilar_dir, filename)
                file_samples = self._load_dissimilar_samples_from_file(file_path)
                samples.extend(file_samples)
        
        self.logger.info(f"加载了 {len(samples)} 个不相似样本")
        return samples
    
    def _load_dissimilar_samples_from_file(self, file_path: str) -> List[IntegratedSample]:
        """
        从PKL文件加载不相似样本
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[IntegratedSample]: 不相似样本列表
        """
        samples = []
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            for pair in data.get('pairs', []):
                # 创建整合样本
                sample = IntegratedSample(
                    source_func=pair.source_func,
                    asm_func=pair.asm_func,
                    similarity_score=pair.similarity_score,
                    sample_type="dissimilar",
                    match_type="dissimilar",
                    confidence=0.8,  # 不相似样本的默认置信度
                    source_signature=str(pair.source_signature),
                    asm_signature=str(pair.asm_signature),
                    metadata={
                        "generation_method": "dissimilar_llm",
                        "source_project": pair.source_signature.project_name,
                        "source_file": pair.source_signature.file_name,
                        "asm_project": pair.asm_signature.project_name,
                        "asm_file": pair.asm_signature.file_name
                    }
                )
                
                samples.append(sample)
                
        except Exception as e:
            self.logger.error(f"加载不相似样本文件失败 {file_path}: {e}")
        
        return samples
    
    def integrate_samples(self, 
                         similar_samples: List[IntegratedSample],
                         dissimilar_samples: List[IntegratedSample],
                         shuffle: bool = True) -> List[IntegratedSample]:
        """
        整合相似和不相似样本
        
        Args:
            similar_samples: 相似样本列表
            dissimilar_samples: 不相似样本列表
            shuffle: 是否打乱顺序
            
        Returns:
            List[IntegratedSample]: 整合后的样本列表
        """
        # 合并样本
        all_samples = similar_samples + dissimilar_samples
        
        # 打乱顺序
        if shuffle:
            random.shuffle(all_samples)
        
        self.logger.info(f"整合完成: 总共 {len(all_samples)} 个样本")
        self.logger.info(f"  - 相似样本: {len(similar_samples)} 个")
        self.logger.info(f"  - 不相似样本: {len(dissimilar_samples)} 个")
        
        return all_samples
    
    def split_dataset(self, 
                     samples: List[IntegratedSample],
                     train_ratio: float = 0.7,
                     test_ratio: float = 0.2,
                     val_ratio: float = 0.1) -> Tuple[List[IntegratedSample], List[IntegratedSample], List[IntegratedSample]]:
        """
        分割数据集
        
        Args:
            samples: 样本列表
            train_ratio: 训练集比例
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            
        Returns:
            Tuple[List[IntegratedSample], List[IntegratedSample], List[IntegratedSample]]: (训练集, 测试集, 验证集)
        """
        # 验证比例
        total_ratio = train_ratio + test_ratio + val_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例总和必须为1.0，当前为{total_ratio}")
        
        # 计算分割点
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        test_size = int(total_samples * test_ratio)
        
        # 分割数据
        train_samples = samples[:train_size]
        test_samples = samples[train_size:train_size + test_size]
        val_samples = samples[train_size + test_size:]
        
        self.logger.info(f"数据集分割完成:")
        self.logger.info(f"  - 训练集: {len(train_samples)} 个样本 ({len(train_samples)/total_samples:.1%})")
        self.logger.info(f"  - 测试集: {len(test_samples)} 个样本 ({len(test_samples)/total_samples:.1%})")
        self.logger.info(f"  - 验证集: {len(val_samples)} 个样本 ({len(val_samples)/total_samples:.1%})")
        
        return train_samples, test_samples, val_samples
    
    def save_samples_to_csv(self, 
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
        
        self.logger.info(f"样本已保存到CSV文件: {filepath}")
        return filepath
    
    def save_samples_to_pkl(self, 
                           samples: List[IntegratedSample],
                           filename: str) -> str:
        """
        保存样本到PKL文件
        
        Args:
            samples: 样本列表
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备保存的数据
        save_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_samples": len(samples),
                "sample_types": {
                    "similar": sum(1 for s in samples if s.sample_type == "similar"),
                    "dissimilar": sum(1 for s in samples if s.sample_type == "dissimilar")
                }
            },
            "samples": samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"样本已保存到PKL文件: {filepath}")
        return filepath
    
    def analyze_dataset(self, samples: List[IntegratedSample]) -> Dict[str, Any]:
        """
        分析数据集
        
        Args:
            samples: 样本列表
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        if not samples:
            return {}
        
        # 基本统计
        total_samples = len(samples)
        similar_count = sum(1 for s in samples if s.sample_type == "similar")
        dissimilar_count = sum(1 for s in samples if s.sample_type == "dissimilar")
        
        # 相似度统计
        similarities = [s.similarity_score for s in samples]
        import numpy as np
        
        stats = {
            "total_samples": total_samples,
            "sample_types": {
                "similar": similar_count,
                "dissimilar": dissimilar_count
            },
            "similarity_stats": {
                "mean": np.mean(similarities),
                "std": np.std(similarities),
                "min": np.min(similarities),
                "max": np.max(similarities),
                "median": np.median(similarities)
            }
        }
        
        # 相似度区间统计
        ranges = {
            "very_high": sum(1 for s in similarities if s >= 0.8),
            "high": sum(1 for s in similarities if 0.6 <= s < 0.8),
            "medium": sum(1 for s in similarities if 0.4 <= s < 0.6),
            "low": sum(1 for s in similarities if 0.2 <= s < 0.4),
            "very_low": sum(1 for s in similarities if s < 0.2)
        }
        stats["similarity_ranges"] = ranges
        
        self.logger.info(f"数据集分析结果:")
        self.logger.info(f"  - 总样本数: {total_samples}")
        self.logger.info(f"  - 相似样本: {similar_count} ({similar_count/total_samples:.1%})")
        self.logger.info(f"  - 不相似样本: {dissimilar_count} ({dissimilar_count/total_samples:.1%})")
        self.logger.info(f"  - 平均相似度: {stats['similarity_stats']['mean']:.3f}")
        self.logger.info(f"  - 相似度范围: {stats['similarity_stats']['min']:.3f} - {stats['similarity_stats']['max']:.3f}")
        
        return stats
