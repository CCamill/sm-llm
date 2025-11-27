"""
将resources/datasets/source_funcs_treesitter目录下的所有json文件中的数据提取出来合并到一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
import os
import json
import pandas as pd
import numpy as np

if __name__ == "__main__":
    source_funcs_root = "resources/datasets/source_funcs_treesitter"
    json_files = []
    for root, dirs, files in os.walk(source_funcs_root):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    all_data = []
    for json_file in json_files:
        try:
            data = json.load(open(json_file, 'r', encoding='utf-8'))
            proj_name = os.path.basename(os.path.dirname(json_file))
            # 提取目标文件名
            relative_path = os.path.relpath(json_file, source_funcs_root)
            parts = relative_path.split(os.sep)
            if len(parts) >= 2:
                target_file = parts[1]  # 假设目录结构为 source_funcs_treesitter/项目名/xxx.json
            else:
                target_file = "unknown"
            file_name = os.path.basename(json_file).split('.')[0]
            for func in data.get("functions", []):
                function_name = func.get("function_name", "")
                source_code = func.get("full_definition", "")
                if len(source_code) < 10:
                    continue  # 跳过源代码过短的函数
                signature = proj_name + '$' + file_name  + '$' + function_name
                all_data.append({
                    "signature": signature,
                    "function_name": function_name,
                    "source_code": source_code,
                })
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    print(f"总共合并了 {len(all_data)} 个函数")
    # 创建DataFrame并保存为合并后的CSV文件
    merged_df = pd.DataFrame(all_data)
    output_csv = "resources/datasets/source_funcs_treesitter/src_funcs.csv"
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"合并完成，输出文件: {output_csv}")