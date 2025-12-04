"""
pandas按照src_func进行分组，然后每组取一个构成一个新的组，如果这个组已经
取完了，那就跳过，比如现在有五个组，每个组有10条数据，那么重组之后有10组，
每个组有5条数据。然后对新的每组数据打乱，按照组的顺序合并成一个新的dataframe，
比如第一组在0-5，第二组在6-10

交错采样
"""
import pandas as pd
import numpy as np
def interleave_groups_compact(df, group_col='func_name'):
    """简洁版本的组交错重组"""
    # 分组并重置索引
    groups = [group.reset_index(drop=True) for _, group in df.groupby(group_col)]
    
    # 找出最大组大小
    max_size = max(len(group) for group in groups)
    
    new_data = []
    
    for i in range(max_size):
        group_data = []
        for j, group in enumerate(groups):
            if i < len(group):
                row = group.iloc[i].copy()
                row['new_group_id'] = i
                row['original_group_rank'] = j
                group_data.append(row)
        
        # 打乱顺序并添加
        np.random.shuffle(group_data)
        new_data.extend(group_data)
    
    return pd.DataFrame(new_data).reset_index(drop=True)

csv_path = "resources/datasets/dataset.csv"
df = pd.read_csv(csv_path)
print("Original DataFrame:")
print(df.head(20))
new_df = interleave_groups_compact(df)
print("Reorganized DataFrame:")
print(new_df.head(20))
new_df.to_csv("resources/datasets/dataset_interleaved.csv", index=False)
