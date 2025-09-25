"""专门用于不相似函数相似度计算的LLM计算器"""

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
from dissimilar_data_loader import DissimilarPair

@dataclass
class DissimilarSimilarityResult:
    """不相似函数相似度结果"""
    source_func: FunctionData
    asm_func: FunctionData
    similarity: float
    confidence: float = 1.0
    reasoning: str = ""

class DissimilarLLMCalculator:
    """专门用于不相似函数的LLM相似度计算器"""
    
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", model_name: str = "qwen2.5-coder:7b"):
        """
        初始化不相似函数LLM计算器
        
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
        self.prompt_template = self._create_dissimilar_prompt_template()
        
        # 创建处理链
        self.chain = self.prompt_template | self.llm | StrOutputParser() | RunnableLambda(self._parse_similarity)
        
    def _create_dissimilar_prompt_template(self) -> ChatPromptTemplate:
        """创建专门用于不相似函数的提示词模板"""
        return ChatPromptTemplate.from_messages([
            ("system", 
            """你是一个专业的代码相似度分析专家。请分析两个来自不同项目/文件/函数的C/C++函数的语义相似度。

                分析要求：
                1. 这两个函数来自不同的项目、文件或具有不同的函数名
                2. 忽略变量名、函数名、结构体名称等标识符的差异
                3. 关注函数的逻辑结构、算法实现、控制流程
                4. 考虑函数的功能目的和实现方式
                5. 输出一个0.00到1.00之间的相似度分数
                6. 只输出数字，不要任何解释

                相似度评分标准：
                - 1.00: 完全相同的逻辑和实现（即使来自不同项目）
                - 0.80-0.99: 高度相似，仅有细微差异
                - 0.60-0.79: 较为相似，主要逻辑相同
                - 0.40-0.59: 部分相似，有共同特征
                - 0.20-0.39: 略有相似，相似度较低
                - 0.00-0.19: 基本不相似或完全不同

                注意：由于这些函数来自不同项目，预期相似度会相对较低，请根据实际语义相似性给出客观评分。"""),
            ("user", """请分析以下两个来自不同项目/文件的函数的相似度：

                源码函数（项目: {source_project}, 文件: {source_file}）：
                ```c
                {source_signature}
                {source_body}
                ```

                汇编函数（项目: {asm_project}, 文件: {asm_file}）：
                ```c
                {asm_signature}
                {asm_body}
                ```

                相似度分数：
                """)
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
                    return 0.3  # 默认中等相似度
        except Exception as e:
            self.logger.error(f"解析相似度分数失败: {e}, 响应: {response}")
            return 0.3
    
    def calculate_dissimilar_similarity(self, pair: DissimilarPair) -> DissimilarSimilarityResult:
        """
        计算不相似函数对的相似度
        
        Args:
            pair: 不相似函数对
            
        Returns:
            DissimilarSimilarityResult: 相似度结果
        """
        try:
            # 准备输入数据
            input_data = {
                "source_project": pair.source_signature.project_name,
                "source_file": pair.source_signature.file_name,
                "source_signature": pair.source_func.signature,
                "source_body": pair.source_func.body,
                "asm_project": pair.asm_signature.project_name,
                "asm_file": pair.asm_signature.file_name,
                "asm_signature": pair.asm_func.signature,
                "asm_body": pair.asm_func.body
            }
            
            # 调用LLM
            similarity = self.chain.invoke(input_data)
            
            return DissimilarSimilarityResult(
                source_func=pair.source_func,
                asm_func=pair.asm_func,
                similarity=similarity
            )
            
        except Exception as e:
            self.logger.error(f"计算不相似函数相似度失败: {e}")
            return DissimilarSimilarityResult(
                source_func=pair.source_func,
                asm_func=pair.asm_func,
                similarity=0.3  # 默认中等相似度
            )
    
    def calculate_batch_dissimilar_similarities(self, 
                                              pairs: List[DissimilarPair],
                                              batch_size: int = 10,
                                              delay: float = 0.2) -> List[DissimilarSimilarityResult]:
        """
        批量计算不相似函数对的相似度
        
        Args:
            pairs: 不相似函数对列表
            batch_size: 批处理大小
            delay: 请求间延迟（秒）
            
        Returns:
            List[DissimilarSimilarityResult]: 相似度结果列表
        """
        results = []
        total_pairs = len(pairs)
        
        self.logger.info(f"开始批量计算不相似函数相似度，共 {total_pairs} 对函数")
        
        for i in range(0, total_pairs, batch_size):
            batch = pairs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_pairs + batch_size - 1) // batch_size
            
            self.logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 个函数对)")
            
            for j, pair in enumerate(batch):
                try:
                    result = self.calculate_dissimilar_similarity(pair)
                    results.append(result)
                    
                    # 更新原始pair的相似度
                    pair.similarity_score = result.similarity
                    
                    if (j + 1) % 5 == 0:
                        self.logger.info(f"  批次 {batch_num}: 已处理 {j + 1}/{len(batch)} 个函数对")
                    
                    # 添加延迟避免请求过于频繁
                    if delay > 0:
                        time.sleep(delay)
                        
                except Exception as e:
                    self.logger.error(f"处理函数对失败: {e}")
                    # 创建默认结果
                    result = DissimilarSimilarityResult(
                        source_func=pair.source_func,
                        asm_func=pair.asm_func,
                        similarity=0.3
                    )
                    results.append(result)
                    pair.similarity_score = 0.3
        
        self.logger.info(f"完成批量相似度计算，共 {len(results)} 个结果")
        return results
    
    def analyze_similarity_distribution(self, results: List[DissimilarSimilarityResult]) -> Dict[str, Any]:
        """
        分析相似度分布
        
        Args:
            results: 相似度结果列表
            
        Returns:
            Dict[str, Any]: 分布统计信息
        """
        similarities = [r.similarity for r in results]
        
        if not similarities:
            return {}
        
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
        
        stats["ranges"] = ranges
        
        self.logger.info(f"不相似函数相似度分布统计:")
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
        
        return stats
