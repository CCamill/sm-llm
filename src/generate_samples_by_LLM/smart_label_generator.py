"""智能标签生成器，基于签名匹配生成所有相似样本"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from data_loader import DataLoader, FunctionData
from function_matcher import FunctionMatcher
from similarity_mapper import SimilarityMapper
from llm_similarity import LLMSimilarityCalculator

@dataclass
class SmartLabelingResult:
    """智能标签生成结果"""
    project_name: str
    total_samples: int
    exact_matches: int
    fuzzy_matches: int
    llm_generated: int
    samples: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class SmartLabelGenerator:
    """智能标签生成器"""
    
    def __init__(self, 
                 source_dir: str,
                 asm_dir: str,
                 output_dir: str,
                 ollama_url: str = "http://127.0.0.1:11434",
                 model_name: str = "qwen2.5-coder:7b"):
        """
        初始化智能标签生成器
        
        Args:
            source_dir: 源码函数数据目录
            asm_dir: 汇编函数数据目录
            output_dir: 输出目录
            ollama_url: Ollama服务URL
            model_name: 模型名称
        """
        self.source_dir = source_dir
        self.asm_dir = asm_dir
        self.output_dir = output_dir
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self.data_loader = DataLoader(source_dir, asm_dir)
        self.function_matcher = FunctionMatcher()
        self.similarity_mapper = SimilarityMapper()
        self.llm_calculator = LLMSimilarityCalculator(ollama_url, model_name)
    
    def generate_all_similar_samples(self, project: str) -> SmartLabelingResult:
        """
        为指定项目生成所有相似样本
        
        Args:
            project: 项目名称
            
        Returns:
            SmartLabelingResult: 标签生成结果
        """
        self.logger.info(f"开始为项目 {project} 生成所有相似样本...")
        
        # 1. 加载项目数据
        source_functions, asm_functions = self.data_loader.load_project_functions(project)
        
        if not source_functions or not asm_functions:
            self.logger.warning(f"项目 {project} 没有足够的数据")
            return SmartLabelingResult(
                project_name=project,
                total_samples=0,
                exact_matches=0,
                fuzzy_matches=0,
                llm_generated=0,
                samples=[],
                statistics={}
            )
        
        # 2. 找到所有匹配的函数对
        matched_pairs = self.function_matcher.find_all_matches(source_functions, asm_functions)
        
        exact_matches = len([p for p in matched_pairs if p.match_type == "exact"])
        fuzzy_matches = len([p for p in matched_pairs if p.match_type == "fuzzy"])
        
        self.logger.info(f"项目 {project}: 找到 {exact_matches} 个精确匹配, {fuzzy_matches} 个模糊匹配")
        
        # 3. 生成所有相似样本
        samples = []
        
        # 精确匹配和模糊匹配的样本
        for pair in matched_pairs:
            sample = {
                "source_function": {
                    "function_id": pair.source_func.function_id,
                    "function_name": pair.source_func.function_name,
                    "signature": pair.source_func.signature,
                    "body": pair.source_func.body,
                    "full_definition": pair.source_func.full_definition,
                    "source_file": pair.source_func.source_file,
                    "file_name": pair.source_func.file_name
                },
                "asm_function": {
                    "function_id": pair.asm_func.function_id,
                    "function_name": pair.asm_func.function_name,
                    "signature": pair.asm_func.signature,
                    "body": pair.asm_func.body,
                    "full_definition": pair.asm_func.full_definition,
                    "source_file": pair.asm_func.source_file,
                    "file_name": pair.asm_func.file_name
                },
                "similarity_label": 1.0 if pair.match_type == "exact" else 0.8,
                "match_type": pair.match_type,
                "confidence": 1.0 if pair.match_type == "exact" else 0.8,
                "metadata": {
                    "generation_method": "signature_matching",
                    "project": project
                }
            }
            samples.append(sample)
        
        # 4. 计算统计信息
        statistics = self._calculate_statistics(samples)
        
        result = SmartLabelingResult(
            project_name=project,
            total_samples=len(samples),
            exact_matches=exact_matches,
            fuzzy_matches=fuzzy_matches,
            llm_generated=0,  # 基于签名匹配，不需要LLM生成
            samples=samples,
            statistics=statistics
        )
        
        self.logger.info(f"项目 {project} 标签生成完成: {len(samples)} 个样本")
        return result
    
    def generate_all_projects_similar_samples(self, projects: List[str] = None) -> List[SmartLabelingResult]:
        """
        为所有项目生成相似样本
        
        Args:
            projects: 项目列表，None表示处理所有项目
            
        Returns:
            List[SmartLabelingResult]: 所有项目的标签生成结果
        """
        if projects is None:
            projects = self.data_loader.get_all_projects()
        
        self.logger.info(f"开始为 {len(projects)} 个项目生成相似样本: {projects}")
        
        results = []
        for project in projects:
            try:
                result = self.generate_all_similar_samples(project)
                results.append(result)
                
                # 保存单个项目的结果
                self._save_project_result(result)
                
            except Exception as e:
                self.logger.error(f"处理项目 {project} 失败: {e}")
                continue
        
        self.logger.info(f"完成所有项目标签生成，共处理 {len(results)} 个项目")
        return results
    
    def _calculate_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算统计信息
        
        Args:
            samples: 样本列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not samples:
            return {}
        
        similarities = [s["similarity_label"] for s in samples]
        import numpy as np
        
        stats = {
            "similarity_stats": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "median": float(np.median(similarities))
            }
        }
        
        return stats
    
    def _save_project_result(self, result: SmartLabelingResult) -> str:
        """
        保存单个项目的结果
        
        Args:
            result: 标签生成结果
            
        Returns:
            str: 保存的文件路径
        """
        output_file = os.path.join(
            self.output_dir, 
            f"{result.project_name}_similar_labels.json"
        )
        
        # 准备保存的数据
        save_data = {
            "metadata": {
                "project_name": result.project_name,
                "generation_timestamp": datetime.now().isoformat(),
                "total_samples": result.total_samples,
                "exact_matches": result.exact_matches,
                "fuzzy_matches": result.fuzzy_matches,
                "llm_generated": result.llm_generated,
                "statistics": result.statistics
            },
            "samples": result.samples
        }
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"项目 {result.project_name} 标签已保存到: {output_file}")
        return output_file
    
    def save_all_results_to_csv(self, results: List[SmartLabelingResult]) -> str:
        """
        保存所有结果到CSV文件
        
        Args:
            results: 所有项目的标签生成结果
            
        Returns:
            str: 保存的文件路径
        """
        output_file = os.path.join(self.output_dir, "all_similar_labels.csv")
        
        # 准备CSV数据
        csv_data = []
        for result in results:
            for sample in result.samples:
                csv_data.append({
                    "source_signature": f"{sample['source_function']['file_name'].split('/')[0] if '/' in sample['source_function']['file_name'] else 'unknown'}+{sample['source_function']['file_name']}+{sample['source_function']['function_name']}",
                    "asm_signature": f"{sample['asm_function']['file_name'].split('/')[0] if '/' in sample['asm_function']['file_name'] else 'unknown'}+{sample['asm_function']['file_name']}+{sample['asm_function']['function_name']}",
                    "similarity_score": sample["similarity_label"],
                    "sample_type": "similar",
                    "match_type": sample["match_type"],
                    "confidence": sample["confidence"],
                    "source_project": sample['source_function']['file_name'].split('/')[0] if '/' in sample['source_function']['file_name'] else 'unknown',
                    "source_file": sample['source_function']['file_name'],
                    "source_function": sample['source_function']['function_name'],
                    "asm_project": sample['asm_function']['file_name'].split('/')[0] if '/' in sample['asm_function']['file_name'] else 'unknown',
                    "asm_file": sample['asm_function']['file_name'],
                    "asm_function": sample['asm_function']['function_name']
                })
        
        # 保存为CSV
        import pandas as pd
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        self.logger.info(f"所有相似样本已保存到CSV文件: {output_file}")
        return output_file
