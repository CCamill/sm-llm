import pandas as pd

if __name__ == "__main__":
    csv_path = "resources/datasets/src_funcs_deduplicated.csv"
    df = pd.read_csv(csv_path)
    print(f"数据集总行数: {len(df)}")
    print("columns: ", df.columns)
    print("前2个函数示例:")
    print(df.head(2))
    print("示例值: ", df[df.columns[0]][0:10].tolist())
    
