import os

ida_path = r"resources/datasets/ida_funcs"
print("ida_funcs共有项目：", len(os.listdir(ida_path)))
csv_num = 0
for root, dirs, files in os.walk(ida_path):
    for file in files:
        if file.endswith(".csv"):
            csv_num += 1
print(f"ida_funcs目录下共有 {csv_num} 个csv文件。")