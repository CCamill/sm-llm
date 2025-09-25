"""不相似样本生成器，专门生成不同签名的函数对并计算相似度"""

import os
import pickle
import csv
import logging
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from dissimilar_data_loader import DissimilarDataLoader, DissimilarPair, FunctionSignature
from llm_similarity import LLMSimilarityCalculator

class DissimilarSampleGenerator:
    """不相似样本生成器"""
    
    def __init__(self, 
                 source_dir: str,
                 asm_dir: str,
                 ollama_url: str = "http://127.0.0.1:11434",
                 model_name: str = "qwen2.5-coder:7b",
                 output_dir: str = "resources/datasets/dissimilar_labels"):
        """
        初始化不相似样本生成器
        
        Args:
            source_dir: 源码函数数据目录
            asm_dir: 汇编函数数据目录
            ollama_url: Ollama服务URL
            model_name: 模型名称
            output_dir: 输出目录
        """
        self.source_dir = source_dir
        self.asm_dir = asm_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.data_loader = DissimilarDataLoader(source_dir, asm_dir)
        self.llm_calculator = LLMSimilarityCalculator(ollama_url, model_name)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dissimilar_samples(self, 
                                  max_pairs: int = 1000,
                                  min_different_projects: int = 1,
                                  batch_size: int = 50,
                                  delay: float = 0.1) -> List[DissimilarPair]:
        """
        生成不相似样本
        
        Args:
            max_pairs: 最大生成对数
            min_different_projects: 最少不同项目数
            batch_size: 批处理大小
            delay: 请求间延迟
            
        Returns:
            List[DissimilarPair]: 不相似样本列表
        """
        self.logger.info("开始生成不相似样本...")
        
        # 1. 加载所有函数数据
        source_functions, asm_functions = self.data_loader.load_all_functions()
        
        if not source_functions or not asm_functions:
            self.logger.error("没有足够的数据生成不相似样本")
            return []
        
        # 2. 生成不相似函数对
        pairs = self.data_loader.generate_dissimilar_pairs(
            source_functions, asm_functions, max_pairs, min_different_projects
        )
        
        if not pairs:
            self.logger.error("未能生成不相似函数对")
            return []
        
        # 3. 分析签名分布
        stats = self.data_loader.analyze_signature_distribution(pairs)
        
        # 4. 使用LLM计算相似度
        self.logger.info("开始使用LLM计算相似度...")
        self._calculate_similarities(pairs, batch_size, delay)
        
        # 5. 分析相似度分布
        self._analyze_similarity_distribution(pairs)
        
        self.logger.info(f"成功生成 {len(pairs)} 个不相似样本")
        return pairs
    
    def _calculate_similarities(self, 
                              pairs: List[DissimilarPair], 
                              batch_size: int = 50,
                              delay: float = 0.1) -> None:
        """
        计算函数对的相似度
        
        Args:
            pairs: 函数对列表
            batch_size: 批处理大小
            delay: 请求间延迟
        """
        total_pairs = len(pairs)
        self.logger.info(f"开始计算 {total_pairs} 个函数对的相似度")
        
        for i in range(0, total_pairs, batch_size):
            batch = pairs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_pairs + batch_size - 1) // batch_size
            
            self.logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 个函数对)")
            
            for j, pair in enumerate(batch):
                try:
                    # 计算相似度
                    similarity_result = self.llm_calculator.calculate_similarity(
                        pair.source_func, pair.asm_func
                    )
                    pair.similarity_score = similarity_result.similarity
                    
                    if (j + 1) % 10 == 0:
                        self.logger.info(f"  批次 {batch_num}: 已处理 {j + 1}/{len(batch)} 个函数对")
                    
                    # 添加延迟
                    if delay > 0:
                        import time
                        time.sleep(delay)
                        
                except Exception as e:
                    self.logger.error(f"计算相似度失败: {e}")
                    pair.similarity_score = 0.0  # 默认相似度
        
        self.logger.info("相似度计算完成")
    
    def _analyze_similarity_distribution(self, pairs: List[DissimilarPair]) -> None:
        """
        分析相似度分布
        
        Args:
            pairs: 函数对列表
        """
        similarities = [pair.similarity_score for pair in pairs]
        
        if not similarities:
            return
        
        import numpy as np
        
        stats = {
            "count": len(similarities),
            "mean": np.mean(similarities),
            "std": np.std(similarities),
            "min": np.min(similarities),
            "max": np.max(similarities),
            "median": np.median(similarities),
            "q25": np.percentile(similarities, 25),
            "q75": np.percentile(similarities, 75)
        }
        
        # 相似度区间统计
        ranges = {
            "very_high": sum(1 for s in similarities if s >= 0.8),
            "high": sum(1 for s in similarities if 0.6 <= s < 0.8),
            "medium": sum(1 for s in similarities if 0.4 <= s < 0.6),
            "low": sum(1 for s in similarities if 0.2 <= s < 0.4),
            "very_low": sum(1 for s in similarities if s < 0.2)
        }
        
        self.logger.info("相似度分布统计:")
        self.logger.info(f"  - 总对数: {stats['count']}")
        self.logger.info(f"  - 平均相似度: {stats['mean']:.3f}")
        self.logger.info(f"  - 标准差: {stats['std']:.3f}")
        self.logger.info(f"  - 范围: {stats['min']:.3f} - {stats['max']:.3f}")
        self.logger.info(f"  - 中位数: {stats['median']:.3f}")
        self.logger.info("相似度区间分布:")
        self.logger.info(f"  - 很高 (≥0.8): {ranges['very_high']} 个")
        self.logger.info(f"  - 高 (0.6-0.8): {ranges['high']} 个")
        self.logger.info(f"  - 中等 (0.4-0.6): {ranges['medium']} 个")
        self.logger.info(f"  - 低 (0.2-0.4): {ranges['low']} 个")
        self.logger.info(f"  - 很低 (<0.2): {ranges['very_low']} 个")
    
    def save_to_pkl(self, pairs: List[DissimilarPair], filename: str = None) -> str:
        """
        保存数据到PKL文件
        
        Args:
            pairs: 函数对列表
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dissimilar_samples_{timestamp}.pkl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备保存的数据
        save_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_pairs": len(pairs),
                "source_dir": self.source_dir,
                "asm_dir": self.asm_dir
            },
            "pairs": pairs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"数据已保存到PKL文件: {filepath}")
        return filepath
    
    def save_to_csv(self, pairs: List[DissimilarPair], filename: str = None) -> str:
        """
        保存数据到CSV文件
        
        Args:
            pairs: 函数对列表
            filename: 文件名
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dissimilar_samples_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 准备CSV数据
        csv_data = []
        for pair in pairs:
            csv_data.append({
                "source_signature": str(pair.source_signature),
                "asm_signature": str(pair.asm_signature),
                "similarity_score": pair.similarity_score,
                "source_project": pair.source_signature.project_name,
                "source_file": pair.source_signature.file_name,
                "source_function": pair.source_signature.function_name,
                "asm_project": pair.asm_signature.project_name,
                "asm_file": pair.asm_signature.file_name,
                "asm_function": pair.asm_signature.function_name
            })
        
        # 保存为CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        self.logger.info(f"数据已保存到CSV文件: {filepath}")
        return filepath
    
    def load_from_pkl(self, filepath: str) -> List[DissimilarPair]:
        """
        从PKL文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            List[DissimilarPair]: 函数对列表
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"从PKL文件加载了 {len(data['pairs'])} 个函数对")
        return data['pairs']
    
    def generate_and_save(self, 
                         max_pairs: int = 1000,
                         min_different_projects: int = 1,
                         batch_size: int = 50,
                         delay: float = 0.1,
                         save_pkl: bool = True,
                         save_csv: bool = True) -> Dict[str, str]:
        """
        生成并保存不相似样本
        
        Args:
            max_pairs: 最大生成对数
            min_different_projects: 最少不同项目数
            batch_size: 批处理大小
            delay: 请求间延迟
            save_pkl: 是否保存PKL文件
            save_csv: 是否保存CSV文件
            
        Returns:
            Dict[str, str]: 保存的文件路径
        """
        # 生成样本
        pairs = self.generate_dissimilar_samples(
            max_pairs, min_different_projects, batch_size, delay
        )
        
        if not pairs:
            self.logger.error("未能生成样本")
            return {}
        
        # 保存文件
        saved_files = {}
        
        if save_pkl:
            pkl_file = self.save_to_pkl(pairs)
            saved_files['pkl'] = pkl_file
        
        if save_csv:
            csv_file = self.save_to_csv(pairs)
            saved_files['csv'] = csv_file
        
        return saved_files
