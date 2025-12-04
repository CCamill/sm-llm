import pandas as pd

if __name__ == "__main__":
    csv_path = "resources/datasets/dataset_interleaved.csv"
    df = pd.read_csv(csv_path)
    print(df.head(-32))
