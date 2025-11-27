"""
ida pro 脚本 需在IDA Pro中运行
导出当前文件的汇编代码到指定的 .s 文件
"""
import idaapi
import idc
import idautils
import os

def main():
    """主函数：导出当前文件的汇编代码"""
    # 获取当前数据库路径
    input_path = idc.get_input_file_path()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = f"{base_name}_ida_export.s"
    
    # 使用兼容的文件选择方法
    try:
        # 方法1: 尝试不同的函数名
        output_path = idc.ask_file(1, "*.s", "选择输出 .s 文件")
    except:
        try:
            # 方法2: 使用 ask_save_file
            output_path = idc.ask_save_file(0, "*.s", "选择输出 .s 文件")
        except:
            try:
                # 方法3: 使用 askfile_c
                output_path = idc.askfile_c(1, "*.s", "选择输出 .s 文件")
            except:
                # 方法4: 直接使用固定路径
                output_path = os.path.join(os.path.dirname(input_path), output_file)
                print(f"使用默认路径: {output_path}")
    
    if not output_path:
        print("用户取消了操作")
        return
    
    print(f"开始导出汇编代码到: {output_path}")
    
    export_assembly_code(output_path)

def export_assembly_code(output_path):
    """导出汇编代码的核心函数"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write(f"; Assembly code exported by IDA Pro\n")
            f.write(f"; Input file: {idc.get_input_file_path()}\n\n")
            
            # 导出所有段
            for seg_ea in idautils.Segments():
                seg_name = idc.get_segm_name(seg_ea)
                seg_start = idc.get_segm_start(seg_ea)
                seg_end = idc.get_segm_end(seg_ea)
                
                f.write(f"; {seg_name} segment ({seg_start:08X}-{seg_end:08X})\n")
                
                # 遍历段中的所有地址
                addr = seg_start
                while addr < seg_end and addr != idc.BADADDR:
                    # 获取汇编指令
                    asm_line = idc.GetDisasm(addr)
                    if asm_line:
                        f.write(f"{addr:08X}    {asm_line}\n")
                    
                    addr = idc.next_head(addr, seg_end)
            
            print(f"导出完成！")
            
    except Exception as e:
        print(f"导出过程中出错: {e}")

# 执行主函数
if __name__ == "__main__":
    main()