"""
从常用第三方库构建构建数据集
包括个各别代码优化
批量多进程处理 .o 文件，保持目录结构
obj/项目/xxx.o -> ida_funcs/项目/xxx.csv
每个 .csv 文件包含函数名、汇编代码等信息
"""
import os
import sys
import csv
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import tempfile
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging(log_prefix: str = "split_func") -> logging.Logger:
    """设置日志配置，将日志保存到resources/logs/{current_time}_{log_prefix}.log。
    
    Args:
        project_root: 项目根目录路径
        log_prefix: 日志文件名前缀
        
    Returns:
        配置好的logger实例
    """
    
    # 生成日志文件名（包含当前时间）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(r"resources/logs", f"{current_time}_{log_prefix}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

logger = setup_logging("split-asm-funcs")

def get_ida_path():
    """获取 IDA Pro 可执行文件路径"""
    # Linux 常见路径
    possible_paths = [
        "/opt/ida/ida64",
        "/usr/local/bin/ida64", 
        "/home/username/ida/ida64",
        "ida64",
        "/home/lab314/sjj/tools/idapro-9.0/idat64"  # 修改为你的实际路径
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果找不到，尝试which命令
    ida_path = shutil.which("ida64") or shutil.which("ida")
    if ida_path:
        return ida_path
    
    raise Exception("未找到IDA Pro可执行文件，请检查安装路径")

def process_single_file_wrapper(args):
    """包装函数用于多进程处理"""
    input_file = args
    return process_single_file(input_file)

def process_single_file(proj_name, opti, input_file):
    """处理单个文件，保持目录结构"""
    try:
        file_name = input_file.split('/')[-1].replace('o', 'csv')
        output_csv_path = os.path.join(r"resources/datasets/ida_funcs/", proj_name, opti, file_name)
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        ida_path = get_ida_path()
        log_file = output_csv_path.replace('.csv', '_ida.log')
        
        # 创建处理脚本 - 关键修复：等待分析和使用正确的API
        script_content = f'''import idc
import idautils
import idaapi
import ida_auto
import csv
import os
import time

# 输出文件路径
output_csv = r"{output_csv_path}"
input_filename = r"{os.path.basename(input_file)}"

print("开始处理:", input_filename)

# 等待IDA完成自动分析
print("等待自动分析完成...")
ida_auto.auto_wait()

# 给分析更多时间
time.sleep(2)

functions_data = []

# 获取所有段中的函数
for seg_ea in idautils.Segments():
    seg_start = idc.get_segm_start(seg_ea)
    seg_end = idc.get_segm_end(seg_ea)
    
    # 遍历段中的所有函数
    for func_ea in idautils.Functions(seg_start, seg_end):
        try:
            func_name = idc.get_func_name(func_ea)
            func_start = func_ea
            func_end = idc.find_func_end(func_ea)
            
            if func_end == idc.BADADDR:
                continue
                
            instructions = []
            addr = func_start
            
            # 收集函数的所有指令
            while addr < func_end and addr != idc.BADADDR:
                asm = idc.GetDisasm(addr)
                if asm:
                    instr_line = hex(addr) + " " + asm
                    instructions.append(instr_line)
                addr = idc.next_head(addr, func_end)
            
            # 准备函数数据
            func_data = {{
                'function_name': func_name,
                'full_define': " | ".join(instructions),
                'start_addr': hex(func_start),
                'source_file': input_filename
            }}
            functions_data.append(func_data)
            print(f"找到函数: {{func_name}} ({{len(instructions)}} 条指令)")
            
        except Exception as e:
            print("处理函数时出错:", str(e))
            continue

print(f"总共找到 {{len(functions_data)}} 个函数")

# 写入CSV文件
try:
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        if functions_data:
            fieldnames = ['function_name', 'full_define', 'start_addr', 'source_file']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(functions_data)
            print("成功写入", len(functions_data), "个函数到", output_csv)
        else:
            print("未找到任何函数")
            # 创建包含错误信息的文件
            with open(output_csv, 'w') as f:
                f.write("function_name,full_define,start_addr,source_file\\n")
                f.write("# No functions found in this file\\n")
except Exception as e:
    print("写入文件时出错:", str(e))
    # 即使出错也创建文件记录错误
    with open(output_csv, 'w') as f:
        f.write("function_name,full_define,start_addr,source_file\\n")
        f.write(f"# Error: {{str(e)}}\\n")

print("处理完成")

# 正确退出IDA
idaapi.qexit(0)
'''
        
        # 将脚本写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_file = f.name
        
        # 执行 IDA 命令 - 增加超时时间
        cmd = [
            get_ida_path(), 
            "-A",
            f"-S{script_file}",
            input_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 清理临时脚本
        if os.path.exists(script_file):
            os.remove(script_file)
        
        if result.returncode == 0 and os.path.exists(output_csv_path):
            return (input_file, True, f"成功: {output_csv_path}")
        else:
            # 输出错误信息
            error_msg = result.stderr or "未知错误"
            return (input_file, False, f"IDA处理失败: {error_msg}")
            
    except subprocess.TimeoutExpired:
        return (input_file, False, "处理超时(300秒)")
    except Exception as e:
        return (input_file, False, f"异常: {str(e)}")

def find_all_o_files(input_dir):
    """递归查找所有.o文件"""
    o_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.o'):
                o_files.append(os.path.join(root, file))
    return o_files

def batch_process_with_structure(input_dir, output_base_dir):
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 {input_dir}")
        return
    
    # 查找所有.o文件
    o_files = find_all_o_files(input_dir)
    
    if not o_files:
        print("未找到 .o 文件")
        return
    print(f"找到 {len(o_files)} 个 .o 文件，开始处理...")
    success_count = 0
    for input_file in o_files:
        input_file, success, message = process_single_file(input_file)
        success_count += int(success)
    
    # 输出统计信息
    print("-" * 60)
    print(f"批量处理完成！")
    print(f"成功: {success_count}/{len(o_files)} 个文件")

def mutiprocess_batch_process_with_structure(input_dir, output_base_dir, max_workers=None):
    """批量处理并保持目录结构"""
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 {input_dir}")
        return
    
    # 查找所有.o文件
    o_files = find_all_o_files(input_dir)
    
    if not o_files:
        print("未找到 .o 文件")
        return
    
    print(f"找到 {len(o_files)} 个 .o 文件，开始批量处理...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"目录结构: obj/项目/xxx.o -> ida_funcs/项目/xxx.csv")
    print("-" * 60)
    
    # 准备任务参数
    tasks = [(f) for f in o_files]
    
    # 多进程处理
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers or multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_single_file_wrapper, task): task for task in tasks}
        
        for future in as_completed(futures):
            input_file, success, message = future.result()
            status = "✓" if success else "✗"
            filename = os.path.basename(input_file)
            print(f"{status} {filename}: {message}")
            if success:
                success_count += 1
    
    # 输出统计信息
    print("-" * 60)
    print(f"批量处理完成！")
    print(f"成功: {success_count}/{len(o_files)} 个文件")
    

def print_output_structure(output_dir, max_depth=3):
    """打印输出目录结构"""
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        if level > max_depth:
            continue
            
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        
        sub_indent = '  ' * (level + 1)
        for file in files:
            if file.endswith('.csv'):
                print(f"{sub_indent}{file}")

if __name__ == "__main__":
    o_dictory = r"/home/lab314/cjw/sm_sog/resources/dataset/gcc/gcc_o"
    src_dictory = r"resources/datasets/source_funcs_treesitter"
    src_projs = os.listdir(src_dictory)

    optimizers = ['-O0', '-O1', '-O2', '-O3', '-Ofast', '-Os']
    len_src_projs = len(src_projs)

    for idx, src_proj in enumerate(src_projs):
        output_csv_proj = os.path.join(r"resources/datasets/ida_funcs/", src_proj)
        if os.path.exists(output_csv_proj):
            logger.info(f"跳过已处理项目: {src_proj}")
            continue
        logger.info(f"Processing {src_proj} {idx}/{len_src_projs}")
        src_proj_path = os.path.join(src_dictory, src_proj)
        if not os.path.isdir(src_proj_path):
            logger.info(f"{src_proj_path} 不是目录，跳过")
            continue
        src_files = os.listdir(src_proj_path)
        src_files = [f.split('.')[0] for f in src_files]
        progress_bar = tqdm(
            src_files,
            desc=f"processing {src_proj}",
            ncols=150,
        )
        success_count = 0
        for file_name in src_files:
            o_files = []
            for opti in optimizers:
                o_proj_dir = os.path.join(o_dictory, src_proj)
                o_file_path = os.path.join(o_proj_dir, opti, file_name+'.o')
                if not os.path.exists(o_file_path):
                    continue
                input_file, success, message = process_single_file(src_proj, opti, o_file_path)
                logger.info(message)
                success_count += int(success)
        logger.info(f"{src_proj} success files: {success_count}")
                    
            

