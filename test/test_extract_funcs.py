#!/usr/bin/env python3
"""
C/C++ 函数信息提取器
使用 tree-sitter 解析 C/C++ 源代码并提取函数信息
tree_sitter_c: 0.24.1
tree-sitter-cpp: 0.23.4
tree-sitter: 0.25.2
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node


class FunctionExtractor:
    """C/C++ 函数信息提取器"""
    
    def __init__(self):
        # 初始化 C 语言解析器
        self.c_language = Language(tsc.language())
        self.c_parser = Parser(self.c_language)
        
        # 初始化 C++ 语言解析器
        self.cpp_language = Language(tscpp.language())
        self.cpp_parser = Parser(self.cpp_language)
    
    def _get_parser(self, file_path: str) -> Parser:
        """根据文件扩展名选择解析器"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.c', '.h']:
            return self.c_parser
        elif ext in ['.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh']:
            return self.cpp_parser
        else:
            # 默认使用 C++ 解析器（兼容性更好）
            return self.cpp_parser
    
    def _get_node_text(self, node: Node, source_code: bytes) -> str:
        """获取节点对应的源代码文本"""
        return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
    
    def _extract_function_name(self, node: Node, source_code: bytes) -> Optional[str]:
        """提取函数名"""
        # 查找 declarator 节点
        declarator = None
        for child in node.children:
            if child.type in ['function_declarator', 'declarator']:
                declarator = child
                break
        
        if declarator is None:
            return None
        
        # 递归查找函数名（处理指针等情况）
        return self._find_identifier(declarator, source_code)
    
    def _find_identifier(self, node: Node, source_code: bytes) -> Optional[str]:
        """递归查找标识符"""
        if node.type == 'identifier':
            return self._get_node_text(node, source_code)
        
        if node.type == 'field_identifier':
            return self._get_node_text(node, source_code)
        
        if node.type == 'destructor_name':
            return self._get_node_text(node, source_code)
        
        if node.type == 'qualified_identifier':
            # 对于 namespace::function 形式，返回完整名称
            return self._get_node_text(node, source_code)
        
        if node.type == 'operator_name':
            return self._get_node_text(node, source_code)
        
        # 优先查找 function_declarator
        for child in node.children:
            if child.type == 'function_declarator':
                result = self._find_identifier(child, source_code)
                if result:
                    return result
        
        # 递归查找
        for child in node.children:
            result = self._find_identifier(child, source_code)
            if result:
                return result
        
        return None
    
    def _extract_signature(self, node: Node, source_code: bytes) -> str:
        """提取函数签名（不包含函数体）"""
        # 查找函数体节点
        body_node = None
        for child in node.children:
            if child.type == 'compound_statement':
                body_node = child
                break
        
        if body_node:
            # 签名是从函数开始到函数体开始（包含左花括号）
            signature_end = body_node.start_byte + 1  # 包含 '{'
            signature = source_code[node.start_byte:signature_end].decode('utf-8', errors='replace')
        else:
            # 如果没有函数体（声明），返回整个节点
            signature = self._get_node_text(node, source_code)
        
        return signature.strip()
    
    def _extract_body(self, node: Node, source_code: bytes) -> str:
        """提取函数体（不包含花括号）"""
        body_node = None
        for child in node.children:
            if child.type == 'compound_statement':
                body_node = child
                break
        
        if body_node:
            # 获取函数体内容（不包含外层花括号）
            body_text = self._get_node_text(body_node, source_code)
            # 移除首尾的花括号
            if body_text.startswith('{'):
                body_text = body_text[1:]
            if body_text.endswith('}'):
                body_text = body_text[:-1]
            return body_text.strip()
        
        return ""
    
    def _find_functions(self, node: Node, source_code: bytes, functions: list, function_id: int) -> int:
        """递归查找所有函数定义"""
        # 函数定义的节点类型
        function_types = [
            'function_definition',      # C/C++ 普通函数定义
        ]
        
        if node.type in function_types:
            func_name = self._extract_function_name(node, source_code)
            if func_name:
                func_info = {
                    "function_id": function_id,
                    "function_name": func_name,
                    "signature": self._extract_signature(node, source_code),
                    "body": self._extract_body(node, source_code)
                }
                functions.append(func_info)
                function_id += 1
        
        # 递归处理子节点
        for child in node.children:
            function_id = self._find_functions(child, source_code, functions, function_id)
        
        return function_id
    
    def extract_from_file(self, file_path: str) -> dict:
        """从单个文件中提取函数信息"""
        file_path = Path(file_path)
        
        # 读取源代码
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        # 选择解析器并解析
        parser = self._get_parser(str(file_path))
        tree = parser.parse(source_code)
        
        # 提取函数
        functions = []
        self._find_functions(tree.root_node, source_code, functions, 1)
        
        # 构建结果
        result = {
            "metadata": {
                "source_file": str(file_path),
                "extraction_timestamp": datetime.now().isoformat(),
                "total_functions": len(functions),
                "extractor": "Tree-sitter extractor"
            },
            "functions": functions
        }
        
        return result
    
    def extract_from_directory(self, dir_path: str, output_dir: str = None, 
                               recursive: bool = True) -> list:
        """从目录中提取所有 C/C++ 文件的函数信息"""
        dir_path = Path(dir_path)
        
        # C/C++ 文件扩展名
        extensions = {'.c', '.h', '.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh'}
        
        # 查找所有文件
        if recursive:
            files = [f for f in dir_path.rglob('*') if f.suffix.lower() in extensions]
        else:
            files = [f for f in dir_path.glob('*') if f.suffix.lower() in extensions]
        
        results = []
        for file_path in files:
            try:
                result = self.extract_from_file(str(file_path))
                results.append(result)
                
                # 如果指定了输出目录，保存到 JSON 文件
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # 生成输出文件名
                    relative_path = file_path.relative_to(dir_path)
                    json_name = str(relative_path).replace(os.sep, '_') + '.json'
                    json_path = output_path / json_name
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"已处理: {file_path} -> {json_path}")
                else:
                    print(f"已处理: {file_path} (找到 {len(result['functions'])} 个函数)")
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='使用 tree-sitter 提取 C/C++ 函数信息'
    )
    parser.add_argument(
        '--input',
        help='输入文件或目录路径',
        default=r"resources/datasets/threelib/curl/src/tool_setopt.c"
    )
    parser.add_argument(
        '-o', '--output',
        help='输出 JSON 文件路径（单文件）或输出目录（目录输入时）'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        default=True,
        help='递归处理子目录（默认开启）'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='不递归处理子目录'
    )
    
    args = parser.parse_args()
    
    extractor = FunctionExtractor()
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 处理单个文件
        result = extractor.extract_from_file(str(input_path))
        
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path) + '.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"提取完成: {output_path}")
        print(f"找到 {result['metadata']['total_functions']} 个函数")
        
    elif input_path.is_dir():
        # 处理目录
        output_dir = args.output if args.output else str(input_path) + '_functions'
        results = extractor.extract_from_directory(
            str(input_path), 
            output_dir, 
            recursive=args.recursive
        )
        
        total_functions = sum(r['metadata']['total_functions'] for r in results)
        print(f"\n提取完成!")
        print(f"处理了 {len(results)} 个文件")
        print(f"共找到 {total_functions} 个函数")
        print(f"输出目录: {output_dir}")
        
    else:
        print(f"错误: 路径不存在 - {args.input}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())