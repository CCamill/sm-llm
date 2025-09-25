"""LLM相似度计算模块，使用Qwen2.5-coder模型"""

import re
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time
import random

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from data_loader import FunctionData

@dataclass
class SimilarityResult:
    """相似度结果"""
    func1: FunctionData
    func2: FunctionData
    similarity: float
    confidence: float = 1.0

class LLMSimilarityCalculator:
    """LLM相似度计算器"""
    
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", model_name: str = "qwen2.5-coder:7b"):
        """
        初始化LLM相似度计算器
        
        Args:
            ollama_url: Ollama服务URL
            model_name: 模型名称
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # 初始化LLM
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            temperature=0.1,  # 降低随机性，提高一致性
            top_p=0.9,
        )
        
        # 创建提示词模板
        self.prompt_template = self._create_prompt_template()
        
        # 创建处理链
        self.chain = self.prompt_template | self.llm | StrOutputParser() | RunnableLambda(self._parse_similarity)
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """创建提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的代码相似度分析专家。请分析两个C/C++函数的语义相似度。

分析要求：
1. 忽略变量名、函数名、结构体名称等标识符的差异
2. 关注函数的逻辑结构、算法实现、控制流程
3. 考虑函数的功能目的和实现方式
4. 输出一个0.00到1.00之间的相似度分数
5. 只输出数字，不要任何解释

相似度评分标准：
- 1.00: 完全相同的逻辑和实现
- 0.80-0.99: 高度相似，仅有细微差异
- 0.60-0.79: 较为相似，主要逻辑相同
- 0.40-0.59: 部分相似，有共同特征
- 0.20-0.39: 略有相似，相似度较低
- 0.00-0.19: 基本不相似或完全不同"""),
            ("user", """请分析以下两个函数的相似度：

函数1：
```c
{func1_signature}
{func1_body}
```

函数2：
```c
{func2_signature}
{func2_body}
```

相似度分数：""")
        ])
    
    def _parse_similarity(self, response: str) -> float:
        """
        解析LLM返回的相似度分数
        
        Args:
            response: LLM响应文本
            
        Returns:
            float: 相似度分数
        """
        try:
            # 提取数字
            numbers = re.findall(r'0\.\d{2}|1\.00|\d+\.\d{1,2}', response)
            if numbers:
                similarity = float(numbers[0])
                # 确保在0-1范围内
                similarity = max(0.0, min(1.0, similarity))
                return similarity
            else:
                # 如果没有找到数字，尝试提取整数
                integers = re.findall(r'\b[0-9]+\b', response)
                if integers:
                    similarity = float(integers[0]) / 100.0
                    similarity = max(0.0, min(1.0, similarity))
                    return similarity
                else:
                    self.logger.warning(f"无法解析相似度分数: {response}")
                    return 0.5  # 默认中等相似度
        except Exception as e:
            self.logger.error(f"解析相似度分数失败: {e}, 响应: {response}")
            return 0.5
    
    def calculate_similarity(self, func1: FunctionData, func2: FunctionData) -> SimilarityResult:
        """
        计算两个函数的相似度
        
        Args:
            func1: 第一个函数
            func2: 第二个函数
            
        Returns:
            SimilarityResult: 相似度结果
        """
        try:
            # 准备输入数据
            input_data = {
                "func1_signature": func1.signature,
                "func1_body": func1.body,
                "func2_signature": func2.signature,
                "func2_body": func2.body
            }
            
            # 调用LLM
            similarity = self.chain.invoke(input_data)
            
            return SimilarityResult(
                func1=func1,
                func2=func2,
                similarity=similarity
            )
            
        except Exception as e:
            self.logger.error(f"计算相似度失败: {e}")
            return SimilarityResult(
                func1=func1,
                func2=func2,
                similarity=0.5  # 默认中等相似度
            )
    
    def calculate_batch_similarities(self, function_pairs: List[Tuple[FunctionData, FunctionData]], 
                                   batch_size: int = 10, delay: float = 0.1) -> List[SimilarityResult]:
        """
        批量计算相似度
        
        Args:
            function_pairs: 函数对列表
            batch_size: 批处理大小
            delay: 请求间延迟（秒）
            
        Returns:
            List[SimilarityResult]: 相似度结果列表
        """
        results = []
        total_pairs = len(function_pairs)
        
        self.logger.info(f"开始批量计算相似度，共 {total_pairs} 对函数")
        
        for i in range(0, total_pairs, batch_size):
            batch = function_pairs[i:i + batch_size]
            self.logger.info(f"处理批次 {i//batch_size + 1}/{(total_pairs + batch_size - 1)//batch_size}")
            
            for func1, func2 in batch:
                result = self.calculate_similarity(func1, func2)
                results.append(result)
                
                # 添加延迟避免请求过于频繁
                if delay > 0:
                    time.sleep(delay)
        
        self.logger.info(f"完成批量相似度计算，共 {len(results)} 个结果")
        return results
    
    def calculate_source_to_source_similarities(self, source_functions: List[FunctionData], 
                                              max_pairs: int = 1000) -> List[SimilarityResult]:
        """
        计算源码函数之间的相似度
        
        Args:
            source_functions: 源码函数列表
            max_pairs: 最大计算对数
            
        Returns:
            List[SimilarityResult]: 相似度结果列表
        """
        # 生成函数对
        pairs = []
        for i in range(len(source_functions)):
            for j in range(i + 1, len(source_functions)):
                pairs.append((source_functions[i], source_functions[j]))
                
                # 限制计算对数
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        
        # 随机打乱顺序
        random.shuffle(pairs)
        
        self.logger.info(f"计算 {len(pairs)} 对源码函数的相似度")
        return self.calculate_batch_similarities(pairs)
    
    def calculate_source_to_asm_similarities(self, source_functions: List[FunctionData], 
                                           asm_functions: List[FunctionData],
                                           source_similarities: Dict[Tuple[str, str], float],
                                           max_pairs: int = 1000) -> List[SimilarityResult]:
        """
        计算源码函数与汇编函数的相似度（基于源码相似度映射）
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            source_similarities: 源码相似度字典 {(func1_key, func2_key): similarity}
            max_pairs: 最大计算对数
            
        Returns:
            List[SimilarityResult]: 相似度结果列表
        """
        results = []
        pair_count = 0
        
        for source_func in source_functions:
            for asm_func in asm_functions:
                if pair_count >= max_pairs:
                    break
                
                # 查找对应的源码相似度
                source_key = (source_func.file_name, source_func.function_name)
                asm_key = (asm_func.file_name, asm_func.function_name)
                
                # 尝试找到最相似的源码函数对
                best_similarity = 0.0
                for (key1, key2), similarity in source_similarities.items():
                    if key1 == source_key or key2 == source_key:
                        # 如果找到相关的源码相似度，使用它作为基础
                        best_similarity = max(best_similarity, similarity)
                
                # 如果没有找到相关的源码相似度，使用默认值
                if best_similarity == 0.0:
                    best_similarity = 0.3  # 默认低相似度
                
                # 应用映射函数（可以根据需要调整）
                mapped_similarity = self._map_source_to_asm_similarity(best_similarity)
                
                result = SimilarityResult(
                    func1=source_func,
                    func2=asm_func,
                    similarity=mapped_similarity
                )
                results.append(result)
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        self.logger.info(f"计算了 {len(results)} 对源码-汇编函数相似度")
        return results
    
    def _map_source_to_asm_similarity(self, source_similarity: float) -> float:
        """
        将源码相似度映射到源码-汇编相似度
        
        Args:
            source_similarity: 源码相似度
            
        Returns:
            float: 映射后的相似度
        """
        # 使用非线性映射，保持相似度的相对关系
        # 可以根据实际需求调整映射函数
        if source_similarity >= 0.8:
            return source_similarity * 0.9  # 高相似度稍微降低
        elif source_similarity >= 0.6:
            return source_similarity * 0.8  # 中等相似度适当降低
        elif source_similarity >= 0.4:
            return source_similarity * 0.7  # 低相似度进一步降低
        else:
            return source_similarity * 0.6  # 很低相似度大幅降低
