"""
将IDA函数和源代码函数的数据进行合并，生成最终的数据集CSV文件。
"""
import pandas as pd

if __name__ == "__main__":
    ida_df = pd.read_csv("resources/datasets/ida_funcs/ida_funcs.csv")
    src_df = pd.read_csv("resources/datasets/source_funcs_treesitter/src_funcs.csv")
    ida_funcs = set(ida_df['signature'].tolist())
    src_funcs = set(src_df['signature'].tolist())
    common_funcs = ida_funcs.intersection(src_funcs)
    
    all_data = []
    for func in common_funcs:
        ida_func_item = ida_df[ida_df['signature'] == func]
        src_func_item = src_df[src_df['signature'] == func]
        func_name = ida_func_item['function_name'].values[0]
        asm_func = ida_func_item['assembly_code'].values[0]
        src_func = src_func_item['source_code'].values[0]
        all_data.append({
            "signature": func,
            "func_name": func_name,
            "asm_func": asm_func,
            "src_func": src_func
        })
    print("total funcs: ", len(all_data))
    merg_df = pd.DataFrame(all_data)
    merg_df.to_csv("resources/datasets/dataset.csv", index=False)
    pass