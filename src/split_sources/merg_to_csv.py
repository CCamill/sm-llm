"""
将resources/datasets/source_funcs_treesitter目录下的所有json文件中的数据提取出来合并到一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
from simhash import Simhash, SimhashIndex
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging(log_prefix: str = "") -> logging.Logger:
    """设置日志配置，将日志保存到resources/logs/{current_time}_{log_prefix}.log。
    
    Args:
        project_root: 项目根目录路径
        log_prefix: 日志文件名前缀
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

logger = setup_logging("merg_src_to_csv")


def get_key_words(source_path, proj_name):
    source_path_list = source_path.split('/')
    proj_namea_idx = source_path_list.index(proj_name)
    key_words = source_path_list[proj_namea_idx + 1: ]

    return list(set(key_words))

def get_jseon_files(source_funcs_root):
    json_files = []
    for root, dirs, files in os.walk(source_funcs_root):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

if __name__ == "__main__":
    merg_step = 20000
    source_funcs_root = "resources/datasets/source_funcs_treesitter"
    output_csv = "resources/datasets/source_funcs_treesitter/src_funcs.csv"
    logger.info(f"源函数JSON文件目录: {source_funcs_root}")
    logger.info(f"输出CSV文件路径: {output_csv}")
    
    json_files = get_jseon_files(source_funcs_root)
    logger.info(f"找到 {len(json_files)} 个JSON文件。")
    all_data = []
    process_bar = tqdm(enumerate(json_files), 
                       total=len(json_files),
                       desc="合并JSON文件", 
                       ncols=100)
    
    logger.info("开始合并JSON文件到CSV...")
    for idx, json_file in process_bar:
        try:
            data = json.load(open(json_file, 'r', encoding='utf-8'))
            proj_name = os.path.basename(os.path.dirname(json_file))
            # 提取目标文件名
            file_name = os.path.basename(json_file).split('.')[0]
            for func in data.get("functions", []):
                function_name = func.get("function_name", "")
                source_code = func.get("body", "")
                function_define = func.get("signature", "")
                if function_name == 'main' or len(source_code.split('\n')) < 5:
                    continue  # main函数和源代码行数过短的函数
                key = proj_name + '$' + file_name  + '$' + function_name
                signature = file_name  + '$' + function_name

                key_words = get_key_words(data["metadata"].get("source_file"), proj_name)
                all_data.append({
                    "key": key,
                    "key_words": key_words,
                    "signature": signature,
                    "function_name": function_name,
                    "source_code": source_code,
                    "hash": Simhash(source_code).value,
                })
            if (idx + 1) % merg_step == 0 or idx == len(json_files) - 1:
                new_data = pd.DataFrame(all_data)
                if not os.path.exists(output_csv):
                    new_data.to_csv(output_csv, index=False, encoding='utf-8')
                else:
                    new_data.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8')
                logger.info(f"已处理文件数: {idx + 1}, 已合并函数数: {len(all_data)}")
                all_data = []  # 清空已保存的数据以节省内存
        except Exception as e:
            logger.info(f"处理文件 {json_file} 时出错: {str(e)}")
    logger.info("JSON文件合并完成。")