"""
根据函数名对数据集进行去重，保留每个函数名的第一个出现。
"""
import pandas as pd


def functions_deduplication(csv_path, dedup_csv_path, key_column="function_name"):
    df = pd.read_csv(csv_path)
    print("前5个函数示例:")
    print(df.head(5))
    print(f"数据集包含 {len(df)} 个函数")

    
    function_names = df[key_column].tolist()
    print(f"数据集包含: {len(set(function_names))} 个不同的函数名")

    new_df = df.drop_duplicates(subset=[key_column], keep="first").reset_index(drop=True)

    print(f"去重后数据集包含 {len(new_df)} 个函数")
    new_df.to_csv(dedup_csv_path, index=False)
    print(f"去重后的数据集已保存到: {dedup_csv_path}")

if __name__ == "__main__":
    csv_path = "resources/datasets/ida_funcs.csv"
    dedup_csv_path = "resources/datasets/ida_funcs_deduplicated.csv"
    functions_deduplication(csv_path, dedup_csv_path, key_column="signature")