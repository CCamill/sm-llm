"""
构建训练和评估数据集
根据resources/datasets/dataset.csv文件，构建适用于训练和评估的数据集
训练集和评估集分别保存在resources/datasets/train_dataset.csv和resources/datasets/eval_dataset.csv
"""
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def build_datasets():
    # 读取完整数据集
    pos_df = pd.read_csv("resources/datasets/dataset.csv")
    pos_df["label"] = 1  # 正样本标签为1
    pos_df.drop('signature', axis=1, inplace=True)
    pos_df.drop('func_name', axis=1, inplace=True)
    neg_list = []
    len_df = len(pos_df)
    for idx in range(len_df):
        item = pos_df.iloc[idx]
        asm_func1 = item['asm_func']
        src_func1 = item['src_func']
        ftype=random.randint(0,len_df-1)
        while ftype==idx:
            ftype=random.randint(0,len_df-1)
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
    
    neg_df = pd.DataFrame(neg_list)

    merge_df = pd.concat([pos_df, neg_df], ignore_index=True)

    df = merge_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 定义比例（例如：60%训练，20%验证，20%测试）
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # 第一次分割：分出训练集
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)

    # 第二次分割：从剩余数据中分出验证集和测试集
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)

    # 保存数据集
    train_df.to_csv("resources/datasets/train_dataset.csv", index=False)
    val_df.to_csv("resources/datasets/eval_dataset.csv", index=False)
    test_df.to_csv("resources/datasets/test_dataset.csv", index=False)

    print(f"训练集大小: {len(train_df)}, 评估集大小: {len(val_df)}, 测试机大小：{len(test_df)}")

if __name__ == "__main__":
    build_datasets()