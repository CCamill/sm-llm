"""
切分数据集为训练集、验证集和测试集，保持原始数据顺序不变
无标签数据集版本，所有样本均为正样本，适合用InfoNCE对比学习方法训练
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
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
    log_file = os.path.join(r"resources/logs", f"{log_prefix}_{current_time}.log")
    
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

logger = setup_logging("split_dataset_no_label")

def split_dataset_preserve_order(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    按照指定比例划分数据集，保持原始数据顺序不变
    
    参数:
        file_path: 数据文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 验证比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    # 读取数据
    
    total_samples = len(df)
    
    logger.info(f"数据集总样本数: {total_samples}")
    logger.info(f"划分比例: 训练集 {train_ratio*100}%, 验证集 {val_ratio*100}%, 测试集 {test_ratio*100}%")
    
    # 计算各集合的样本数量
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size  # 确保总数正确
    
    logger.info(f"训练集样本数: {train_size}")
    logger.info(f"验证集样本数: {val_size}")
    logger.info(f"测试集样本数: {test_size}")
    
    # 按照原始顺序划分
    train_set = df.iloc[:train_size]
    val_set = df.iloc[train_size:train_size + val_size]
    test_set = df.iloc[train_size + val_size:]
    
    return train_set, val_set, test_set

def verify_split(original, train, val, test):
    """
    验证划分结果的正确性
    """
    
    # 验证样本总数
    total_original = len(original)
    total_split = len(train) + len(val) + len(test)
    
    logger.info("\n=== 验证结果 ===")
    logger.info(f"原始数据集样本数: {total_original}")
    logger.info(f"划分后总样本数: {len(train)} + {len(val)} + {len(test)} = {total_split}")
    logger.info(f"样本数是否一致: {total_original == total_split}")
    
    # 验证顺序是否保持
    is_ordered = True
    # 检查训练集是否是原始数据的前train_size个
    if not original.iloc[:len(train)].equals(train):
        is_ordered = False
    
    # 检查验证集是否是接下来的val_size个
    if not original.iloc[len(train):len(train)+len(val)].equals(val):
        is_ordered = False
    
    # 检查测试集是否是剩余的部分
    if not original.iloc[len(train)+len(val):].equals(test):
        is_ordered = False
    
    logger.info(f"原始顺序是否保持: {is_ordered}")
    
    return total_original == total_split and is_ordered

# 主程序
if __name__ == "__main__":
    # 文件路径
    file_path = "resources/datasets/dataset_interleaved.csv"
    train_set_path = "resources/datasets/dataset_train_no_label.csv"
    val_set_path = "resources/datasets/dataset_eval_no_label.csv"
    test_set_path = "resources/datasets/dataset_test_no_label.csv"
    
    try:
        # 划分数据集
        origin_df = pd.read_csv(file_path)
        train_set, val_set, test_set = split_dataset_preserve_order(
            origin_df, 
            train_ratio=0.7, 
            val_ratio=0.2, 
            test_ratio=0.1
        )
        
        train_set.to_csv(train_set_path, index=False)
        logger.info(f"训练集已保存到: {train_set_path}")
        val_set.to_csv(val_set_path, index=False)
        logger.info(f"验证集已保存到: {val_set_path}")
        test_set.to_csv(test_set_path, index=False)
        logger.info(f"测试集已保存到: {test_set_path}")
        
        
        # 验证划分结果
        verification_passed = verify_split(
            origin_df,
            train_set,
            val_set,
            test_set
        )
        
        if verification_passed:
            logger.info("\n✅ 数据集划分完成且验证通过！")
        else:
            logger.info("\n❌ 数据集划分验证失败！")
            
    except FileNotFoundError:
        logger.info("请检查文件路径是否正确")
    except Exception as e:
        logger.info(f"发生错误: {e}")