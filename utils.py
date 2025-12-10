"""通用工具函数模块。

本模块包含项目中使用的通用工具函数，如日志设置、文件处理等。
"""

import logging
import os
from datetime import datetime


def setup_logging(log_prefix: str = "split_func") -> logging.Logger:
    """设置日志配置，将日志保存到resources/logs/{current_time}_{log_prefix}.log。
    
    Args:
        project_root: 项目根目录路径
        log_prefix: 日志文件名前缀
        
    Returns:
        配置好的logger实例
    """
    
    # 生成日志文件名（包含当前时间）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(r"resources/logs", f"{current_time}_{log_prefix}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger


def should_process_file(filename: str, supported_extensions: set) -> bool:
    """检查文件是否应该被处理。
    
    过滤规则：
    - 仅处理扩展名在 supported_extensions 内的文件
    - 过滤以 "test_" 开头的测试文件
    
    Args:
        filename: 文件名
        supported_extensions: 支持的文件扩展名集合
        
    Returns:
        如果文件应该被处理返回True，否则返回False
    """
    basename = os.path.basename(filename)
    if basename.startswith("test_"):
        return False
    _, ext = os.path.splitext(basename)
    return ext in supported_extensions


def count_total_files(input_base: str, supported_extensions: set) -> int:
    """预先统计需要处理的文件总数。
    
    Args:
        input_base: 输入基础目录
        supported_extensions: 支持的文件扩展名集合
        
    Returns:
        需要处理的文件总数
    """
    total_count = 0
    for root_dir, _, files in os.walk(input_base):
        for name in files:
            if should_process_file(name, supported_extensions):
                total_count += 1
    return total_count


def map_input_to_output(input_base: str, output_base: str, file_path: str) -> str:
    """将输入源码路径映射到输出JSON路径。
    
    Args:
        input_base: 输入基础目录
        output_base: 输出基础目录
        file_path: 输入文件路径
        
    Returns:
        输出JSON文件路径
        
    Raises:
        ValueError: 当无法从路径解析库名时
    """
    rel_path = os.path.relpath(file_path, input_base)
    parts = rel_path.split(os.sep)
    if not parts:
        raise ValueError(f"无法从路径解析库名: {file_path}")
    lib_name = parts[0]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_base, lib_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{base_name}.json")
    return out_file
