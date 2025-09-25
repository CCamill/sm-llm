"""使用 Tree-sitter 的函数提取（C/C++）。

依赖: tree_sitter (Python binding), 语言语法库 tree-sitter-c / tree-sitter-cpp
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, should_process_file, count_total_files, map_input_to_output

SUPPORTED_EXTS = {".c", ".cc", ".cpp", ".cxx", ".C"}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from tree_sitter import Language, Parser
    TREESITTER_AVAILABLE = True
except Exception:
    TREESITTER_AVAILABLE = False


class TreeSitterFunctionExtractor:
    def __init__(self):
        self.logger = setup_logging(PROJECT_ROOT)
        if not TREESITTER_AVAILABLE:
            self.logger.warning("未找到 tree_sitter，请安装以使用该提取器")
        self.parser = None
        self.lang = None

    def _ensure_language(self, language: str):
        if not TREESITTER_AVAILABLE:
            raise RuntimeError("tree_sitter 不可用")
        if self.parser is not None and self.lang is not None:
            return
        # 优先路径1：使用 tree_sitter_languages 内置解析器
        try:
            from tree_sitter_languages import get_parser  # type: ignore
            lang_name = "c" if language == "c" else "cpp"
            self.parser = get_parser(lang_name)
            self.lang = None
            return
        except Exception:
            pass

        # 优先路径2：加载现有 .so（如用户预编译或之前已构建）
        lang_obj = None
        if lang_obj is None:
            build_dir = os.path.join(PROJECT_ROOT, "build", "treesitter")
            so_path = os.path.join(build_dir, "languages.so")
            if os.path.isfile(so_path):
                try:
                    lang_obj = Language(so_path, language)
                except Exception:
                    lang_obj = None

        # 优先路径3：在支持时尝试动态构建（部分 tree_sitter 版本不提供 build_library）
        if lang_obj is None and hasattr(Language, "build_library"):
            build_dir = os.path.join(PROJECT_ROOT, "build", "treesitter")
            os.makedirs(build_dir, exist_ok=True)
            so_path = os.path.join(build_dir, "languages.so")
            try:
                Language.build_library(
                    so_path,
                    [
                        os.path.join(PROJECT_ROOT, "third_party", "tree-sitter-c"),
                        os.path.join(PROJECT_ROOT, "third_party", "tree-sitter-cpp"),
                    ],
                )
                lang_obj = Language(so_path, language)
            except Exception:
                lang_obj = None

        if lang_obj is None:
            raise RuntimeError(
                "无法加载 Tree-sitter 语言。请安装 tree_sitter_languages 或提供预编译的 languages.so。"
            )

        parser = Parser()
        parser.set_language(lang_obj)
        self.lang = lang_obj
        self.parser = parser

    def extract_functions_from_text(self, code_text: str, language: str) -> List[Dict[str, Any]]:
        self._ensure_language(language)
        tree = self.parser.parse(code_text.encode("utf-8"))
        root = tree.root_node

        results: List[Dict[str, Any]] = []
        func_id = 1

        def slice_text(node):
            return code_text[node.start_byte: node.end_byte]

        def extract_function_name(node):
            """从函数定义节点中提取函数名称"""
            # 查找函数声明器 (function_declarator)
            for child in node.children:
                if child.type == "function_declarator":
                    # 在函数声明器中查找标识符
                    for grandchild in child.children:
                        if grandchild.type == "identifier":
                            return slice_text(grandchild)
                        # 处理指针函数的情况 (如 int (*func)(int))
                        elif grandchild.type == "parenthesized_declarator":
                            for ggchild in grandchild.children:
                                if ggchild.type == "pointer_declarator":
                                    for gggchild in ggchild.children:
                                        if gggchild.type == "function_declarator":
                                            for ggggchild in gggchild.children:
                                                if ggggchild.type == "identifier":
                                                    return slice_text(ggggchild)
            return "unknown_function"

        def visit(node):
            nonlocal func_id
            # C: function_definition; C++: function_definition / function_declaration + compound_statement
            type_name = node.type
            if type_name in ("function_definition",):
                snippet = slice_text(node)
                signature = snippet.split('\n', 1)[0].strip()
                
                # 提取函数名称
                function_name = extract_function_name(node)
                
                results.append({
                    "function_id": func_id,
                    "function_name": function_name,
                    "signature": signature,
                    "body": snippet[len(signature):].lstrip('\n'),
                    "full_definition": snippet,
                })
                func_id += 1
            for child in node.children:
                visit(child)

        visit(root)
        return results


def _resolve_bases(project_root: str) -> Tuple[str, str]:
    candidates = [
        os.path.join(project_root, "resources", "datasets", "threelib"),
        os.path.join(project_root, "src", "resources", "datasets", "threelib"),
    ]
    input_base = None
    for cand in candidates:
        if os.path.isdir(cand):
            input_base = cand
            break
    if input_base is None:
        raise FileNotFoundError(
            "未找到输入目录 resources/datasets/threelib（或src/resources/datasets/threelib）"
        )

    output_base = os.path.join(project_root, "resources", "datasets", "source_funcs_treesitter")
    os.makedirs(output_base, exist_ok=True)
    return input_base, output_base


def main():
    logger = setup_logging(PROJECT_ROOT)
    input_base, output_base = _resolve_bases(PROJECT_ROOT)
    extractor = TreeSitterFunctionExtractor()

    total_files = 0
    total_funcs = 0

    for root_dir, _, files in os.walk(input_base):
        # 过滤构建目录，避免缺失或中间产物
        if "/build/" in root_dir.replace("\\", "/"):
            continue
        for name in files:
            if not should_process_file(name, SUPPORTED_EXTS):
                continue
            file_path = os.path.join(root_dir, name)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_text = f.read()
                language = "cxx" if os.path.splitext(name)[1].lower() in {".cc", ".cpp", ".cxx", ".C"} else "c"
                functions = extractor.extract_functions_from_text(code_text, language)
                if not functions:
                    continue
                timestamp = datetime.now().isoformat()
                json_output = {
                    "metadata": {
                        "source_file": os.path.relpath(file_path, PROJECT_ROOT),
                        "extraction_timestamp": timestamp,
                        "total_functions": len(functions),
                        "extractor": "Tree-sitter extractor",
                    },
                    "functions": functions,
                }
                out_file = map_input_to_output(input_base, output_base, file_path)
                with open(out_file, "w", encoding="utf-8") as of:
                    json.dump(json_output, of, ensure_ascii=False, indent=2)
                total_files += 1
                total_funcs += len(functions)
                logger.info(
                    f"✅ {os.path.relpath(file_path, input_base)} -> {os.path.relpath(out_file, output_base)} (提取{len(functions)}个函数)"
                )
            except Exception as e:
                logger.error(f"处理文件失败: {file_path} | 错误: {e}")

    logger.info(
        f"完成。共处理文件 {total_files} 个，累计提取函数 {total_funcs} 个。"
    )


if __name__ == "__main__":
    main()


