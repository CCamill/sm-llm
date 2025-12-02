"""
将IDA函数和源代码函数的数据进行合并，生成最终的数据集CSV文件。
"""
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    ida_df = pd.read_csv("resources/datasets/ida_funcs_deduplicated.csv")
    src_df = pd.read_csv("resources/datasets/source_funcs_treesitter/src_funcs_deduplicated.csv")
    ida_keys = set(ida_df['key'].tolist())
    src_keys = set(src_df['key'].tolist())
    common_keys = ida_keys.intersection(src_keys)
    
    all_data = []
    tqdm_iter = tqdm(enumerate(common_keys), 
                     total=len(common_keys),
                     desc="合并IDA函数和源代码函数", 
                     ncols=100)
    for idx, key in tqdm_iter:
        func_name = key.split('$')[-1]
        ida_func_items = ida_df[ida_df['key'] == key]    # 同一个函数有不同优化级别的版本
        src_func_item = src_df[src_df['key'] == key]
        asm_funcs = ida_func_items['assembly_code'].values
        src_func = src_func_item['source_code'].values[0]
        if len(ida_func_items) > 1:
            pass
        for i in range(len(ida_func_items)):
            asm_func = asm_funcs[i]
            all_data.append({
                "key": key,
                "func_name": func_name,
                "asm_func": asm_func,
                "src_func": src_func
            })

    print("total pairs: ", len(all_data))
    merg_df = pd.DataFrame(all_data)
    merg_df.to_csv("resources/datasets/dataset.csv", index=False)
    pass