"""
构建训练和评估数据集
根据resources/datasets/dataset.csv文件，构建适用于训练和评估的数据集
训练集和评估集分别保存在resources/datasets/train_dataset.csv和resources/datasets/eval_dataset.csv
"""
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
from tqdm import tqdm

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

logger = setup_logging("build_train_eval_test_dataset")


def build_datasets(pos_df, out_path):
    # 读取完整数据集
    pos_df["label"] = 1  # 正样本标签为1
    pos_df.drop('func_name', axis=1, inplace=True)

    neg_list = []
    len_df = len(pos_df)
    logger.info(f"正在为数据集 {out_path} 生成负样本...")

    pb = tqdm(total=len_df, ncols=150)
    for idx in range(len_df):
        item = pos_df.iloc[idx]
        key = item['key']
        asm_func1 = item['asm_func']
        src_func1 = item['src_func']
        
        ftype=random.randint(0, len_df-1)
        while ftype==idx:
            ftype=random.randint(0,len_df-1)
            if pos_df.iloc[ftype]['key'] == key:
                ftype=idx
        negative_item = pos_df.iloc[ftype]
        src_func2 = negative_item['src_func']
        asm_func2 = negative_item['asm_func']
        neg_list.append({
            "asm_func": asm_func1,
            "src_func": src_func2,
            "label": 0
        })
        neg_list.append({
            "asm_func": asm_func2,
            "src_func": src_func1,
            "label": 0
        })
        pb.update(1)
    pb.close()
    
    neg_df = pd.DataFrame(neg_list)

    pos_df.drop('key', axis=1, inplace=True)

    merge_df = pd.concat([pos_df, neg_df], ignore_index=True)

    df = merge_df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    logger.info(f"数据集已保存到: {out_path}")
    logger.info(f"数据集 {out_path} 包含 {len(df)} 条数据，其中正样本 {len(pos_df)} 条，负样本 {len(neg_df)} 条。")

def split_dataset_by_function(dataset_path):
    """
    按函数名将数据集划分为训练集、评估集和测试集，确保同一函数的不同版本不会出现在不同的数据集中。
    """
    # 读取数据
    df = pd.read_csv(dataset_path)
    
    # 按function列分组
    groups = df.groupby('key')
    
    # 获取所有组名
    group_names = list(groups.groups.keys())
    
    # 第一步：先分出测试集（20%的组）
    train_eval_groups, test_groups = train_test_split(
        group_names, 
        test_size=0.2, 
        random_state=42
    )
    
    # 第二步：从剩余的80%中分出评估集（10%/80% = 12.5%得到总体的10%）
    train_groups, eval_groups = train_test_split(
        train_eval_groups, 
        test_size=0.125,  # 0.125 * 0.8 = 0.1
        random_state=42
    )
    
    # 根据分组名称获取对应的数据
    train_data = pd.concat([groups.get_group(name) for name in train_groups])
    eval_data = pd.concat([groups.get_group(name) for name in eval_groups])
    test_data = pd.concat([groups.get_group(name) for name in test_groups])
    
    # 打印统计信息
    print(f"总数据量: {len(df)}")
    print(f"总组数: {len(group_names)}")
    print(f"训练集: {len(train_data)} 条数据 ({len(train_groups)} 个组)")
    print(f"评估集: {len(eval_data)} 条数据 ({len(eval_groups)} 个组)")
    print(f"测试集: {len(test_data)} 条数据 ({len(test_groups)} 个组)")

    return train_data, eval_data, test_data

def main(dataset_path, train_dataset_path, eval_dataset_path, test_dataset_path):
    # 读取训练集和评估集
    train_data, eval_data, test_data = split_dataset_by_function(
        dataset_path=dataset_path
        )
    build_datasets(train_data, train_dataset_path)
    build_datasets(eval_data, eval_dataset_path)
    build_datasets(test_data, test_dataset_path)

if __name__ == "__main__":
    dataset_path = 'resources/datasets/dataset.csv'
    train_dataset_path = 'resources/datasets/train_dataset.csv'
    eval_dataset_path = 'resources/datasets/eval_dataset.csv'
    test_dataset_path = 'resources/datasets/test_dataset.csv'
    # 执行函数
    main(dataset_path, train_dataset_path, eval_dataset_path, test_dataset_path)