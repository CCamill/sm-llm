"""不相似样本数据加载器，用于生成跨项目/文件/函数的函数对"""

import os
import json
import random
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import logging

from data_loader import DataLoader, FunctionData

@dataclass
class FunctionSignature:
    """函数签名数据结构"""
    project_name: str
    file_name: str
    function_name: str
    
    def __str__(self):
        return f"{self.project_name}+{self.file_name}+{self.function_name}"
    
    def __hash__(self):
        return hash((self.project_name, self.file_name, self.function_name))
    
    def __eq__(self, other):
        if not isinstance(other, FunctionSignature):
            return False
        return (self.project_name == other.project_name and 
                self.file_name == other.file_name and 
                self.function_name == other.function_name)

@dataclass
class DissimilarPair:
    """不相似函数对"""
    source_func: FunctionData
    asm_func: FunctionData
    source_signature: FunctionSignature
    asm_signature: FunctionSignature
    similarity_score: float = 0.0

class DissimilarDataLoader:
    """不相似样本数据加载器"""
    
    def __init__(self, source_dir: str, asm_dir: str):
        """
        初始化不相似样本数据加载器
        
        Args:
            source_dir: 源码函数数据目录
            asm_dir: 汇编函数数据目录
        """
        self.source_dir = source_dir
        self.asm_dir = asm_dir
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(source_dir, asm_dir)
    
    def load_all_functions(self) -> Tuple[List[FunctionData], List[FunctionData]]:
        """
        加载所有项目的源码函数和汇编函数
        
        Returns:
            Tuple[List[FunctionData], List[FunctionData]]: (所有源码函数, 所有汇编函数)
        """
        all_source_functions = []
        all_asm_functions = []
        
        projects = self.data_loader.get_all_projects()
        self.logger.info(f"发现 {len(projects)} 个项目: {projects}")
        
        for project in projects:
            source_functions, asm_functions = self.data_loader.load_project_functions(project)
            all_source_functions.extend(source_functions)
            all_asm_functions.extend(asm_functions)
        
        self.logger.info(f"总共加载了 {len(all_source_functions)} 个源码函数, {len(all_asm_functions)} 个汇编函数")
        return all_source_functions, all_asm_functions
    
    def create_function_signature(self, func: FunctionData) -> FunctionSignature:
        """
        创建函数签名
        
        Args:
            func: 函数数据
            
        Returns:
            FunctionSignature: 函数签名
        """
        return FunctionSignature(
            project_name=func.project_name,
            file_name=func.file_name,
            function_name=func.function_name
        )
    
    def generate_dissimilar_pairs(self, 
                                source_functions: List[FunctionData],
                                asm_functions: List[FunctionData],
                                max_pairs: int = 1000,
                                min_different_projects: int = 1) -> List[DissimilarPair]:
        """
        生成不相似的函数对
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            max_pairs: 最大生成对数
            min_different_projects: 最少不同项目数
            
        Returns:
            List[DissimilarPair]: 不相似函数对列表
        """
        pairs = []
        used_source_signatures = set()
        used_asm_signatures = set()
        
        # 创建函数签名映射
        source_signatures = {}
        asm_signatures = {}
        
        for func in source_functions:
            sig = self.create_function_signature(func)
            source_signatures[sig] = func
        
        for func in asm_functions:
            sig = self.create_function_signature(func)
            asm_signatures[sig] = func
        
        self.logger.info(f"创建了 {len(source_signatures)} 个源码函数签名, {len(asm_signatures)} 个汇编函数签名")
        
        # 生成不相似对
        source_sig_list = list(source_signatures.keys())
        asm_sig_list = list(asm_signatures.keys())
        
        # 随机打乱顺序
        random.shuffle(source_sig_list)
        random.shuffle(asm_sig_list)
        
        pair_count = 0
        attempts = 0
        max_attempts = max_pairs * 10  # 防止无限循环
        
        while pair_count < max_pairs and attempts < max_attempts:
            attempts += 1
            
            # 随机选择源码和汇编函数签名
            source_sig = random.choice(source_sig_list)
            asm_sig = random.choice(asm_sig_list)
            
            # 检查是否已经使用过
            if source_sig in used_source_signatures or asm_sig in used_asm_signatures:
                continue
            
            # 检查是否为不相似对（不同签名）
            if source_sig == asm_sig:
                continue
            
            # 检查项目分布
            if min_different_projects > 1:
                # 统计已使用对中不同项目的数量
                used_projects = set()
                for pair in pairs:
                    used_projects.add(pair.source_signature.project_name)
                    used_projects.add(pair.asm_signature.project_name)
                
                # 如果当前对不会增加项目多样性，跳过
                if len(used_projects) >= min_different_projects:
                    current_projects = used_projects.copy()
                    current_projects.add(source_sig.project_name)
                    current_projects.add(asm_sig.project_name)
                    if len(current_projects) == len(used_projects):
                        continue
            
            # 创建不相似对
            pair = DissimilarPair(
                source_func=source_signatures[source_sig],
                asm_func=asm_signatures[asm_sig],
                source_signature=source_sig,
                asm_signature=asm_sig
            )
            
            pairs.append(pair)
            used_source_signatures.add(source_sig)
            used_asm_signatures.add(asm_sig)
            pair_count += 1
            
            if pair_count % 100 == 0:
                self.logger.info(f"已生成 {pair_count} 个不相似对")
        
        self.logger.info(f"成功生成 {len(pairs)} 个不相似函数对")
        return pairs
    
    def analyze_signature_distribution(self, pairs: List[DissimilarPair]) -> Dict[str, Any]:
        """
        分析签名分布
        
        Args:
            pairs: 不相似函数对列表
            
        Returns:
            Dict[str, Any]: 分布统计信息
        """
        source_projects = set()
        asm_projects = set()
        source_files = set()
        asm_files = set()
        
        for pair in pairs:
            source_projects.add(pair.source_signature.project_name)
            asm_projects.add(pair.asm_signature.project_name)
            source_files.add(pair.source_signature.file_name)
            asm_files.add(pair.asm_signature.file_name)
        
        stats = {
            "total_pairs": len(pairs),
            "source_projects": len(source_projects),
            "asm_projects": len(asm_projects),
            "source_files": len(source_files),
            "asm_files": len(asm_files),
            "project_overlap": len(source_projects.intersection(asm_projects)),
            "file_overlap": len(source_files.intersection(asm_files))
        }
        
        self.logger.info(f"签名分布统计:")
        self.logger.info(f"  - 总对数: {stats['total_pairs']}")
        self.logger.info(f"  - 源码项目数: {stats['source_projects']}")
        self.logger.info(f"  - 汇编项目数: {stats['asm_projects']}")
        self.logger.info(f"  - 项目重叠数: {stats['project_overlap']}")
        self.logger.info(f"  - 文件重叠数: {stats['file_overlap']}")
        
        return stats
    
    def filter_pairs_by_similarity(self, 
                                 pairs: List[DissimilarPair],
                                 min_similarity: float = 0.0,
                                 max_similarity: float = 1.0) -> List[DissimilarPair]:
        """
        根据相似度过滤函数对
        
        Args:
            pairs: 函数对列表
            min_similarity: 最小相似度
            max_similarity: 最大相似度
            
        Returns:
            List[DissimilarPair]: 过滤后的函数对列表
        """
        filtered_pairs = []
        
        for pair in pairs:
            if min_similarity <= pair.similarity_score <= max_similarity:
                filtered_pairs.append(pair)
        
        self.logger.info(f"过滤后保留 {len(filtered_pairs)} 个函数对 (相似度范围: {min_similarity}-{max_similarity})")
        return filtered_pairs
