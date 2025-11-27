import pandas as pd

if __name__ == "__main__":
    csv_path = "resources/datasets/dataset.csv"
    df = pd.read_csv(csv_path)
    print(f"数据集包含 {len(df)} 个函数")
    print("前5个函数示例:")
    print(df.head(5))
