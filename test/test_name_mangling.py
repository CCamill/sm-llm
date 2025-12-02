import cxxfilt
import re

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

# 测试示例
test_functions = [
    "_ZL32_hb_buffer_serialize_glyphs_jsonP11hb_buffer_tjjPcjPjP9hb_font_t27hb_buffer_serialize_flags_t",
    "_ZNSt8ios_base4InitC1Ev",
    "normal_function",
    "?MyFunction@@YAHH@Z",
    "_ZN11hb_buffer_t8add_infoERK15hb_glyph_info_t",
    "hb_buffer_serialize"
]

for func in test_functions:
    print(f"原始: {func}")
    print(f"是否修饰: {is_mangled(func)}")
    if is_mangled(func):
        demangled = demangle_function(func)
        print(f"还原: {demangled}")
    print("-" * 50)

csv_path = r"resources/datasets/ida_funcs/tectonic-typesetting_____tectonic/-O0/hb-buffer-serialize.csv"

