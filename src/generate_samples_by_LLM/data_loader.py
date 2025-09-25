"""数据加载器模块，用于读取源码函数和汇编函数数据"""

import os
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class FunctionData:
    """函数数据结构"""
    function_id: int
    function_name: str
    signature: str
    body: str
    full_definition: str
    source_file: str
    project_name: str
    file_name: str

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, source_dir: str, asm_dir: str):
        """
        初始化数据加载器
        
        Args:
            source_dir: 源码函数数据目录
            asm_dir: 汇编函数数据目录
        """
        self.source_dir = source_dir
        self.asm_dir = asm_dir
        self.logger = logging.getLogger(__name__)
        
    def load_project_functions(self, project_name: str) -> Tuple[List[FunctionData], List[FunctionData]]:
        """
        加载指定项目的源码函数和汇编函数数据
        
        Args:
            project_name: 项目名称
            
        Returns:
            Tuple[List[FunctionData], List[FunctionData]]: (源码函数列表, 汇编函数列表)
        """
        source_functions = []
        asm_functions = []
        
        # 加载源码函数
        source_project_dir = os.path.join(self.source_dir, project_name)
        if os.path.exists(source_project_dir):
            source_functions = self._load_functions_from_directory(
                source_project_dir, project_name, "source"
            )
        
        # 加载汇编函数
        asm_project_dir = os.path.join(self.asm_dir, project_name)
        if os.path.exists(asm_project_dir):
            asm_functions = self._load_functions_from_directory(
                asm_project_dir, project_name, "asm"
            )
        
        self.logger.info(f"项目 {project_name}: 加载了 {len(source_functions)} 个源码函数, {len(asm_functions)} 个汇编函数")
        return source_functions, asm_functions
    
    def _load_functions_from_directory(self, directory: str, project_name: str, data_type: str) -> List[FunctionData]:
        """
        从目录中加载函数数据
        
        Args:
            directory: 数据目录
            project_name: 项目名称
            data_type: 数据类型 ("source" 或 "asm")
            
        Returns:
            List[FunctionData]: 函数数据列表
        """
        functions = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                file_functions = self._load_functions_from_file(file_path, project_name, data_type)
                functions.extend(file_functions)
        
        return functions
    
    def _load_functions_from_file(self, file_path: str, project_name: str, data_type: str) -> List[FunctionData]:
        """
        从JSON文件中加载函数数据
        
        Args:
            file_path: 文件路径
            project_name: 项目名称
            data_type: 数据类型
            
        Returns:
            List[FunctionData]: 函数数据列表
        """
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 获取源文件路径
            source_file = data.get('metadata', {}).get('source_file', '')
            
            for func_data in data.get('functions', []):
                if data_type == "source":
                    function = FunctionData(
                        function_id=func_data.get('function_id', 0),
                        function_name=func_data.get('function_name', ''),
                        signature=func_data.get('signature', ''),
                        body=func_data.get('body', ''),
                        full_definition=func_data.get('full_definition', ''),
                        source_file=source_file,
                        project_name=project_name,
                        file_name=file_name
                    )
                else:  # asm
                    function = FunctionData(
                        function_id=func_data.get('function_id', 0),
                        function_name=func_data.get('name', ''),  # 汇编函数使用 'name' 字段
                        signature=func_data.get('signature', ''),
                        body=func_data.get('body', ''),
                        full_definition=func_data.get('full_definition', ''),
                        source_file=source_file,
                        project_name=project_name,
                        file_name=file_name
                    )
                
                functions.append(function)
                
        except Exception as e:
            self.logger.error(f"加载文件失败 {file_path}: {e}")
        
        return functions
    
    def get_all_projects(self) -> List[str]:
        """
        获取所有可用的项目列表
        
        Returns:
            List[str]: 项目名称列表
        """
        projects = set()
        
        # 从源码目录获取项目
        if os.path.exists(self.source_dir):
            projects.update(os.listdir(self.source_dir))
        
        # 从汇编目录获取项目
        if os.path.exists(self.asm_dir):
            projects.update(os.listdir(self.asm_dir))
        
        return sorted(list(projects))
    
    def find_matching_functions(self, source_functions: List[FunctionData], 
                              asm_functions: List[FunctionData]) -> List[Tuple[FunctionData, FunctionData]]:
        """
        找到匹配的源码函数和汇编函数对
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            
        Returns:
            List[Tuple[FunctionData, FunctionData]]: 匹配的函数对列表
        """
        matches = []
        
        # 创建汇编函数的查找字典，按文件名和函数名索引
        asm_dict = {}
        for asm_func in asm_functions:
            key = (asm_func.file_name, asm_func.function_name)
            asm_dict[key] = asm_func
        
        # 查找匹配的源码函数
        for source_func in source_functions:
            key = (source_func.file_name, source_func.function_name)
            if key in asm_dict:
                matches.append((source_func, asm_dict[key]))
        
        self.logger.info(f"找到 {len(matches)} 个匹配的函数对")
        return matches
    
    def get_non_matching_functions(self, source_functions: List[FunctionData], 
                                 asm_functions: List[FunctionData]) -> Tuple[List[FunctionData], List[FunctionData]]:
        """
        获取不匹配的源码函数和汇编函数
        
        Args:
            source_functions: 源码函数列表
            asm_functions: 汇编函数列表
            
        Returns:
            Tuple[List[FunctionData], List[FunctionData]]: (不匹配的源码函数, 不匹配的汇编函数)
        """
        matches = self.find_matching_functions(source_functions, asm_functions)
        matched_source_ids = {match[0].function_id for match in matches}
        matched_asm_ids = {match[1].function_id for match in matches}
        
        non_matching_source = [f for f in source_functions if f.function_id not in matched_source_ids]
        non_matching_asm = [f for f in asm_functions if f.function_id not in matched_asm_ids]
        
        return non_matching_source, non_matching_asm
