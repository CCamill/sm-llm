"""
将resources/datasets/ida_funcs目录下的所有csv文件合并为一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
import os
import json
import pandas as pd
import numpy as np
import re
import cxxfilt

def is_mangled(name):
    """判断函数名是否经过名称修饰"""
    # C++ 名称修饰的常见特征
    patterns = [
        r'^_Z',                    # Itanium C++ ABI
        r'^\?',                    # Microsoft修饰
        r'^__Z',                   # 某些系统的Itanium变体
    ]
    
    for pattern in patterns:
        if re.match(pattern, name):
            return True
    return False

def demangle_function(name):
    """还原名称修饰的函数名"""
    try:
        return cxxfilt.demangle(name)
    except:
        return name  # 如果还原失败，返回原名称

def extract_function_name_advanced(demangled_name):
    """高级函数名提取，处理各种复杂情况"""
    
    # 移除尾部的参数列表
    if '(' in demangled_name and demangled_name.endswith(')'):
        # 找到最后一个括号对
        stack = []
        for i, char in enumerate(demangled_name):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    start = stack.pop()
                    if not stack:  # 最外层的括号对
                        base_name = demangled_name[:start].strip()
                        break
        else:
            base_name = demangled_name.split('(')[0].strip()
    else:
        base_name = demangled_name
    
    # 提取函数名部分（去掉返回类型）
    patterns = [
        # 匹配: 返回类型 函数名
        r'^[a-zA-Z_][a-zA-Z0-9_:]+\s+([a-zA-Z_][a-zA-Z0-9_]+)$',
        # 匹配: 函数名 (无返回类型的情况，如构造函数)
        r'^([a-zA-Z_][a-zA-Z0-9_]+)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, base_name.strip())
        if match:
            return match.group(1)
    
    # 如果以上都不匹配，尝试分割空格取最后一部分
    parts = base_name.strip().split()
    if parts:
        return parts[-1]  # 通常函数名在最后
    
    return base_name



def get_csv_files(source_funcs_root):
    csv_files = []
    for root, dirs, files in os.walk(source_funcs_root):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def merage_ida_csv():
    ida_funcs_root = "resources/datasets/ida_funcs"
    csv_files = get_csv_files(ida_funcs_root)
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment='#')
            if df.empty:
                continue
            proj_name = os.path.basename(os.path.dirname(csv_file))
            # 提取目标文件名
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

def merage_ida_csv_with_opti():
    ida_funcs_root = "resources/datasets/ida_funcs"
    csv_files = get_csv_files(ida_funcs_root)
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment='#')
            if df.empty:
                continue
            opti = re.search(r'-(O[\dsfastgz]+)', csv_file)
            opti_level = opti.group(1) if opti else "O0"
            if opti:
                path_list = csv_file.split(r'/')
                proj_name = path_list[-3]
            else:
                proj_name = os.path.basename(os.path.dirname(csv_file))
            # 提取目标文件名
            file_name = os.path.basename(csv_file).split('.')[0]
            
            
            for _, row in df.iterrows():
                function_name = row.get("function_name", "")
                if is_mangled(function_name):
                    function_name = demangle_function(function_name)
                    function_name = extract_function_name_advanced(function_name)
                assembly_code = row.get("full_define", "")
                if assembly_code==np.nan:
                    continue  # 跳过汇编代码过短的函数
                if len(assembly_code.split('|')) < 7:
                    continue  # 跳过汇编代码过短的函数

                key = proj_name + '$' + file_name  + '$' + function_name
                signature = function_name + '$' + opti_level
                all_data.append({
                    "key": key,
                    "signature": signature,
                    "function_name": function_name,
                    "assembly_code": assembly_code,
                    "opti_level": opti_level,
                })
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
    print(f"总共合并了 {len(all_data)} 个函数")
    # 创建DataFrame并保存为合并后的CSV文件
    merged_df = pd.DataFrame(all_data)
    output_csv = "resources/datasets/ida_funcs.csv"
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"合并完成，输出文件: {output_csv}")


if __name__ == "__main__":
    merage_ida_csv_with_opti()