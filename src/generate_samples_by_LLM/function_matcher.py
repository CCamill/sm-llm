"""函数匹配逻辑模块，用于识别对应的函数对"""

import logging
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from data_loader import FunctionData

@dataclass
class FunctionMatch:
    """函数匹配结果"""
    source_func: FunctionData
    asm_func: FunctionData
    match_confidence: float
    match_type: str  # "exact", "fuzzy", "manual"

class FunctionMatcher:
    """函数匹配器"""
    
    def __init__(self):
        """初始化函数匹配器"""
        self.logger = logging.getLogger(__name__)
    
    def find_exact_matches(self, source_functions: List[FunctionData], 
                          asm_functions: List[FunctionData]) -> List[FunctionMatch]:
        """
        查找精确匹配的函数对
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            
        Returns:
            List[FunctionMatch]: 精确匹配结果列表
        """
        matches = []
        
        # 创建汇编函数的查找字典
        asm_dict = {}
        for asm_func in asm_functions:
            key = (asm_func.file_name, asm_func.function_name)
            asm_dict[key] = asm_func
        
        # 查找精确匹配
        for source_func in source_functions:
            key = (source_func.file_name, source_func.function_name)
            if key in asm_dict:
                asm_func = asm_dict[key]
                match = FunctionMatch(
                    source_func=source_func,
                    asm_func=asm_func,
                    match_confidence=1.0,
                    match_type="exact"
                )
                matches.append(match)
                self.logger.debug(f"精确匹配: {source_func.function_name} -> {asm_func.function_name}")
        
        self.logger.info(f"找到 {len(matches)} 个精确匹配的函数对")
        return matches
    
    def find_fuzzy_matches(self, source_functions: List[FunctionData], 
                          asm_functions: List[FunctionData],
                          exact_matches: List[FunctionMatch]) -> List[FunctionMatch]:
        """
        查找模糊匹配的函数对
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            exact_matches: 已找到的精确匹配列表
            
        Returns:
            List[FunctionMatch]: 模糊匹配结果列表
        """
        matches = []
        
        # 获取已匹配的函数ID
        matched_source_ids = {match.source_func.function_id for match in exact_matches}
        matched_asm_ids = {match.asm_func.function_id for match in exact_matches}
        
        # 过滤未匹配的函数
        unmatched_source = [f for f in source_functions if f.function_id not in matched_source_ids]
        unmatched_asm = [f for f in asm_functions if f.function_id not in matched_asm_ids]
        
        # 按文件名分组
        source_by_file = self._group_by_file(unmatched_source)
        asm_by_file = self._group_by_file(unmatched_asm)
        
        # 在相同文件内查找模糊匹配
        for file_name in source_by_file:
            if file_name in asm_by_file:
                file_matches = self._find_file_fuzzy_matches(
                    source_by_file[file_name], 
                    asm_by_file[file_name]
                )
                matches.extend(file_matches)
        
        self.logger.info(f"找到 {len(matches)} 个模糊匹配的函数对")
        return matches
    
    def _group_by_file(self, functions: List[FunctionData]) -> Dict[str, List[FunctionData]]:
        """
        按文件名分组函数
        
        Args:
            functions: 函数列表
            
        Returns:
            Dict[str, List[FunctionData]]: 按文件名分组的函数字典
        """
        groups = {}
        for func in functions:
            if func.file_name not in groups:
                groups[func.file_name] = []
            groups[func.file_name].append(func)
        return groups
    
    def _find_file_fuzzy_matches(self, source_functions: List[FunctionData], 
                                asm_functions: List[FunctionData]) -> List[FunctionMatch]:
        """
        在单个文件内查找模糊匹配
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            
        Returns:
            List[FunctionMatch]: 模糊匹配结果列表
        """
        matches = []
        
        for source_func in source_functions:
            best_match = None
            best_confidence = 0.0
            
            for asm_func in asm_functions:
                confidence = self._calculate_name_similarity(
                    source_func.function_name, 
                    asm_func.function_name
                )
                
                if confidence > best_confidence and confidence > 0.7:  # 相似度阈值
                    best_confidence = confidence
                    best_match = asm_func
            
            if best_match:
                match = FunctionMatch(
                    source_func=source_func,
                    asm_func=best_match,
                    match_confidence=best_confidence,
                    match_type="fuzzy"
                )
                matches.append(match)
                self.logger.debug(f"模糊匹配: {source_func.function_name} -> {best_match.function_name} (置信度: {best_confidence:.2f})")
        
        return matches
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        计算函数名相似度
        
        Args:
            name1: 第一个函数名
            name2: 第二个函数名
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not name1 or not name2:
            return 0.0
        
        # 转换为小写进行比较
        name1 = name1.lower()
        name2 = name2.lower()
        
        # 完全匹配
        if name1 == name2:
            return 1.0
        
        # 包含关系
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # 计算编辑距离相似度
        distance = self._levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        
        if max_len == 0:
            return 0.0
        
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离
        
        Args:
            s1: 第一个字符串
            s2: 第二个字符串
            
        Returns:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_all_matches(self, source_functions: List[FunctionData], 
                        asm_functions: List[FunctionData]) -> List[FunctionMatch]:
        """
        查找所有匹配的函数对
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            
        Returns:
            List[FunctionMatch]: 所有匹配结果列表
        """
        # 查找精确匹配
        exact_matches = self.find_exact_matches(source_functions, asm_functions)
        
        # 查找模糊匹配
        fuzzy_matches = self.find_fuzzy_matches(source_functions, asm_functions, exact_matches)
        
        # 合并结果
        all_matches = exact_matches + fuzzy_matches
        
        self.logger.info(f"总共找到 {len(all_matches)} 个匹配的函数对")
        return all_matches
    
    def get_unmatched_functions(self, source_functions: List[FunctionData], 
                               asm_functions: List[FunctionData],
                               matches: List[FunctionMatch]) -> Tuple[List[FunctionData], List[FunctionData]]:
        """
        获取未匹配的函数
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            matches: 匹配结果列表
            
        Returns:
            Tuple[List[FunctionData], List[FunctionData]]: (未匹配的源码函数, 未匹配的汇编函数)
        """
        matched_source_ids = {match.source_func.function_id for match in matches}
        matched_asm_ids = {match.asm_func.function_id for match in matches}
        
        unmatched_source = [f for f in source_functions if f.function_id not in matched_source_ids]
        unmatched_asm = [f for f in asm_functions if f.function_id not in matched_asm_ids]
        
        self.logger.info(f"未匹配的源码函数: {len(unmatched_source)} 个")
        self.logger.info(f"未匹配的汇编函数: {len(unmatched_asm)} 个")
        
        return unmatched_source, unmatched_asm
    
    def validate_matches(self, matches: List[FunctionMatch]) -> List[FunctionMatch]:
        """
        验证匹配结果的有效性
        
        Args:
            matches: 匹配结果列表
            
        Returns:
            List[FunctionMatch]: 验证后的匹配结果列表
        """
        valid_matches = []
        
        for match in matches:
            # 检查函数是否来自相同文件
            if match.source_func.file_name != match.asm_func.file_name:
                self.logger.warning(f"文件不匹配: {match.source_func.file_name} vs {match.asm_func.file_name}")
                continue
            
            # 检查函数名是否合理相似
            if match.match_type == "fuzzy" and match.match_confidence < 0.5:
                self.logger.warning(f"相似度过低: {match.source_func.function_name} vs {match.asm_func.function_name}")
                continue
            
            valid_matches.append(match)
        
        self.logger.info(f"验证后保留 {len(valid_matches)} 个有效匹配")
        return valid_matches
