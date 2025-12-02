import pandas as pd

if __name__ == "__main__":
    csv_path = "resources/datasets/dataset.csv"
    df = pd.read_csv(csv_path)
    print("columns: ", df.columns)
    print("前2个函数示例:")
    print(df.head(2))
    
