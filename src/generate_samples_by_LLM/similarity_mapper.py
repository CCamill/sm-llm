"""相似度映射模块，将源码相似度映射到源码-汇编相似度"""

import logging
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

from data_loader import FunctionData
from llm_similarity import SimilarityResult

@dataclass
class MappedSimilarity:
    """映射后的相似度结果"""
    source_func: FunctionData
    asm_func: FunctionData
    original_similarity: float
    mapped_similarity: float
    mapping_method: str

class SimilarityMapper:
    """相似度映射器"""
    
    def __init__(self):
        """初始化相似度映射器"""
        self.logger = logging.getLogger(__name__)
    
    def map_source_to_asm_similarities(self, 
                                     source_similarities: List[SimilarityResult],
                                     source_functions: List[FunctionData],
                                     asm_functions: List[FunctionData],
                                     mapping_strategy: str = "semantic_preservation") -> List[MappedSimilarity]:
        """
        将源码相似度映射到源码-汇编相似度
        
        Args:
            source_similarities: 源码函数相似度列表
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            mapping_strategy: 映射策略
            
        Returns:
            List[MappedSimilarity]: 映射后的相似度列表
        """
        # 创建源码相似度字典
        source_sim_dict = self._create_similarity_dict(source_similarities)
        
        # 创建函数查找字典
        source_func_dict = {f.function_id: f for f in source_functions}
        asm_func_dict = {f.function_id: f for f in asm_functions}
        
        mapped_similarities = []
        
        for source_func in source_functions:
            for asm_func in asm_functions:
                # 查找相关的源码相似度
                related_similarities = self._find_related_similarities(
                    source_func, source_sim_dict, source_func_dict
                )
                
                if related_similarities:
                    # 计算映射后的相似度
                    mapped_sim = self._calculate_mapped_similarity(
                        related_similarities, mapping_strategy
                    )
                else:
                    # 如果没有找到相关相似度，使用默认值
                    mapped_sim = 0.3
                
                mapped_similarity = MappedSimilarity(
                    source_func=source_func,
                    asm_func=asm_func,
                    original_similarity=related_similarities[0] if related_similarities else 0.0,
                    mapped_similarity=mapped_sim,
                    mapping_method=mapping_strategy
                )
                mapped_similarities.append(mapped_similarity)
        
        self.logger.info(f"映射了 {len(mapped_similarities)} 个源码-汇编相似度")
        return mapped_similarities
    
    def _create_similarity_dict(self, similarities: List[SimilarityResult]) -> Dict[Tuple[int, int], float]:
        """
        创建相似度字典
        
        Args:
            similarities: 相似度结果列表
            
        Returns:
            Dict[Tuple[int, int], float]: 相似度字典
        """
        sim_dict = {}
        for sim in similarities:
            key = (sim.func1.function_id, sim.func2.function_id)
            sim_dict[key] = sim.similarity
            # 添加反向键
            reverse_key = (sim.func2.function_id, sim.func1.function_id)
            sim_dict[reverse_key] = sim.similarity
        return sim_dict
    
    def _find_related_similarities(self, 
                                 source_func: FunctionData,
                                 source_sim_dict: Dict[Tuple[int, int], float],
                                 source_func_dict: Dict[int, FunctionData]) -> List[float]:
        """
        查找与给定源码函数相关的相似度
        
        Args:
            source_func: 源码函数
            source_sim_dict: 源码相似度字典
            source_func_dict: 源码函数字典
            
        Returns:
            List[float]: 相关相似度列表
        """
        related_similarities = []
        
        for (id1, id2), similarity in source_sim_dict.items():
            if id1 == source_func.function_id or id2 == source_func.function_id:
                related_similarities.append(similarity)
        
        return related_similarities
    
    def _calculate_mapped_similarity(self, 
                                   related_similarities: List[float],
                                   mapping_strategy: str) -> float:
        """
        计算映射后的相似度
        
        Args:
            related_similarities: 相关相似度列表
            mapping_strategy: 映射策略
            
        Returns:
            float: 映射后的相似度
        """
        if not related_similarities:
            return 0.3
        
        # 计算平均相似度
        avg_similarity = np.mean(related_similarities)
        
        if mapping_strategy == "linear":
            return self._linear_mapping(avg_similarity)
        elif mapping_strategy == "semantic_preservation":
            return self._semantic_preservation_mapping(avg_similarity)
        elif mapping_strategy == "conservative":
            return self._conservative_mapping(avg_similarity)
        elif mapping_strategy == "aggressive":
            return self._aggressive_mapping(avg_similarity)
        else:
            return self._default_mapping(avg_similarity)
    
    def _linear_mapping(self, similarity: float) -> float:
        """
        线性映射
        
        Args:
            similarity: 原始相似度
            
        Returns:
            float: 映射后的相似度
        """
        # 简单的线性缩放
        return similarity * 0.8
    
    def _semantic_preservation_mapping(self, similarity: float) -> float:
        """
        语义保持映射（推荐）
        
        Args:
            similarity: 原始相似度
            
        Returns:
            float: 映射后的相似度
        """
        # 使用非线性映射保持语义关系
        if similarity >= 0.9:
            return similarity * 0.95  # 高相似度保持
        elif similarity >= 0.7:
            return similarity * 0.85  # 中等相似度适当降低
        elif similarity >= 0.5:
            return similarity * 0.75  # 低相似度进一步降低
        else:
            return similarity * 0.6   # 很低相似度大幅降低
    
    def _conservative_mapping(self, similarity: float) -> float:
        """
        保守映射
        
        Args:
            similarity: 原始相似度
            
        Returns:
            float: 映射后的相似度
        """
        # 保守的映射，降低所有相似度
        return similarity * 0.7
    
    def _aggressive_mapping(self, similarity: float) -> float:
        """
        激进映射
        
        Args:
            similarity: 原始相似度
            
        Returns:
            float: 映射后的相似度
        """
        # 激进的映射，保持更多相似度
        return min(1.0, similarity * 1.1)
    
    def _default_mapping(self, similarity: float) -> float:
        """
        默认映射
        
        Args:
            similarity: 原始相似度
            
        Returns:
            float: 映射后的相似度
        """
        return self._semantic_preservation_mapping(similarity)
    
    def create_similarity_matrix(self, 
                               source_functions: List[FunctionData],
                               asm_functions: List[FunctionData],
                               mapped_similarities: List[MappedSimilarity]) -> np.ndarray:
        """
        创建相似度矩阵
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            mapped_similarities: 映射后的相似度列表
            
        Returns:
            np.ndarray: 相似度矩阵
        """
        # 创建函数ID到索引的映射
        source_id_to_idx = {f.function_id: i for i, f in enumerate(source_functions)}
        asm_id_to_idx = {f.function_id: i for i, f in enumerate(asm_functions)}
        
        # 初始化相似度矩阵
        matrix = np.zeros((len(source_functions), len(asm_functions)))
        
        # 填充相似度矩阵
        for mapped_sim in mapped_similarities:
            source_idx = source_id_to_idx.get(mapped_sim.source_func.function_id, -1)
            asm_idx = asm_id_to_idx.get(mapped_sim.asm_func.function_id, -1)
            
            if source_idx >= 0 and asm_idx >= 0:
                matrix[source_idx, asm_idx] = mapped_sim.mapped_similarity
        
        return matrix
    
    def analyze_similarity_distribution(self, 
                                      mapped_similarities: List[MappedSimilarity]) -> Dict[str, Any]:
        """
        分析相似度分布
        
        Args:
            mapped_similarities: 映射后的相似度列表
            
        Returns:
            Dict[str, Any]: 分布统计信息
        """
        similarities = [ms.mapped_similarity for ms in mapped_similarities]
        
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
            "high_similarity": sum(1 for s in similarities if s >= 0.8),
            "medium_similarity": sum(1 for s in similarities if 0.5 <= s < 0.8),
            "low_similarity": sum(1 for s in similarities if 0.2 <= s < 0.5),
            "very_low_similarity": sum(1 for s in similarities if s < 0.2)
        }
        
        stats["ranges"] = ranges
        
        self.logger.info(f"相似度分布统计: 平均={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
        self.logger.info(f"高相似度(≥0.8): {ranges['high_similarity']} 个")
        self.logger.info(f"中等相似度(0.5-0.8): {ranges['medium_similarity']} 个")
        self.logger.info(f"低相似度(0.2-0.5): {ranges['low_similarity']} 个")
        self.logger.info(f"很低相似度(<0.2): {ranges['very_low_similarity']} 个")
        
        return stats
