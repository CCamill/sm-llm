"""
将resources/datasets/ida_funcs目录下的所有csv文件合并为一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
import os
import json
import pandas as pd
import numpy as np


if __name__ == "__main__":
    ida_funcs_root = "resources/datasets/ida_funcs"
    csv_files = []
    for root, dirs, files in os.walk(ida_funcs_root):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment='#')
            if df.empty:
                continue
            proj_name = os.path.basename(os.path.dirname(csv_file))
            # 提取目标文件名
            relative_path = os.path.relpath(csv_file, ida_funcs_root)
            parts = relative_path.split(os.sep)
            if len(parts) >= 2:
                target_file = parts[1]  # 假设目录结构为 ida_funcs/项目名/xxx.csv
            else:
                target_file = "unknown"
            file_name = os.path.basename(csv_file).split('.')[0]
            
            for _, row in df.iterrows():
                function_name = row.get("function_name", "")
                assembly_code = row.get("full_define", "")
                if assembly_code==np.nan:
                    continue  # 跳过汇编代码过短的函数
                if len(assembly_code) < 10:
                    continue  # 跳过汇编代码过短的函数
                start_addr = row.get("start_addr", "")
                signature = proj_name + '$' + file_name  + '$' + function_name
                all_data.append({
                    "signature": signature,
                    "function_name": function_name,
                    "assembly_code": assembly_code,
                    "start_addr": start_addr,
                })
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
    
    print(f"总共合并了 {len(all_data)} 个函数")
    # 创建DataFrame并保存为合并后的CSV文件
    merged_df = pd.DataFrame(all_data)
    output_csv = "resources/datasets/ida_funcs/ida_funcs.csv"
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"合并完成，输出文件: {output_csv}")
