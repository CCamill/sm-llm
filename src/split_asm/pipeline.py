"""C/C++ 源码 → 目标文件(.o) → 反汇编(.s) → 提取函数(JSON) 管线。

功能：
- 遍历 `resources/datasets/threelib` 下的 C/C++ 文件
- 编译为 .o（支持本机架构；若安装了交叉工具链，也支持 armhf/aarch64 等）
- 使用 objdump 反汇编为 .s
- 解析 .s 中的函数，输出与 split_functions 类似的 JSON 结构

输出目录：
- 对象文件：`resources/datasets/obj`
- 汇编文件：`resources/datasets/asm`
- 函数JSON：`resources/datasets/asm_funcs`

路径映射示例：
- 输入：resources/datasets/threelib/curl/src/slist_wc.c
- 本机架构 .o：resources/datasets/obj/curl/src/slist_wc.o
- 本机架构 .s：resources/datasets/asm/curl/src/slist_wc.s
- 本机架构 .json：resources/datasets/asm_funcs/curl/src/slist_wc.json
- 交叉架构（如 aarch64）将输出到带架构前缀的子目录：
  - resources/datasets/obj/aarch64/curl/src/slist_wc.o
  - resources/datasets/asm/aarch64/curl/src/slist_wc.s
  - resources/datasets/asm_funcs/aarch64/curl/src/slist_wc.json
"""

import os
import sys
import json
import shlex
import subprocess
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, should_process_file, map_input_to_output  # type: ignore


SUPPORTED_EXTS = {".c", ".cc", ".cpp", ".cxx", ".C"}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class Toolchain:
    arch: str
    gcc: str
    gxx: str
    objdump: str
    cflags: List[str]
    cxxflags: List[str]
    objdump_flags: List[str]


def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)


def detect_toolchains() -> List[Toolchain]:
    chains: List[Toolchain] = []
    native_arch = platform.machine() or "unknown"

    # 本机工具链（假设 gcc/objdump 可用）
    gcc = which("gcc") or which("clang")
    gxx = which("g++") or which("clang++")
    objdump = which("objdump")
    if gcc and gxx and objdump:
        objdump_flags = ["-d"]
        if native_arch in {"x86_64", "amd64"}:
            # 使用 Intel 汇编语法便于阅读
            objdump_flags = ["-d", "-Mintel,x86-64", "--no-show-raw-insn"]
        
        # 基础编译参数
        base_cflags = ["-c", "-O2", "-fno-asynchronous-unwind-tables"]
        base_cxxflags = ["-c", "-O2", "-fno-asynchronous-unwind-tables", "-std=c++17", "-fno-exceptions", "-fno-rtti"]
        
        chains.append(Toolchain(
            arch=native_arch,
            gcc=gcc,
            gxx=gxx,
            objdump=objdump,
            cflags=base_cflags,
            cxxflags=base_cxxflags,
            objdump_flags=objdump_flags,
        ))

    # 交叉：ARM 32 (armhf)
    if which("arm-linux-gnueabihf-gcc") and which("arm-linux-gnueabihf-g++") and which("arm-linux-gnueabihf-objdump"):
        chains.append(Toolchain(
            arch="armhf",
            gcc="arm-linux-gnueabihf-gcc",
            gxx="arm-linux-gnueabihf-g++",
            objdump="arm-linux-gnueabihf-objdump",
            cflags=["-c", "-O2"],
            cxxflags=["-c", "-O2", "-std=c++17", "-fno-exceptions", "-fno-rtti"],
            objdump_flags=["-d"],
        ))

    # 交叉：ARM64 (aarch64)
    if which("aarch64-linux-gnu-gcc") and which("aarch64-linux-gnu-g++") and which("aarch64-linux-gnu-objdump"):
        chains.append(Toolchain(
            arch="aarch64",
            gcc="aarch64-linux-gnu-gcc",
            gxx="aarch64-linux-gnu-g++",
            objdump="aarch64-linux-gnu-objdump",
            cflags=["-c", "-O2"],
            cxxflags=["-c", "-O2", "-std=c++17", "-fno-exceptions", "-fno-rtti"],
            objdump_flags=["-d"],
        ))

    return chains


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def map_src_to_obj_and_asm(input_base: str, obj_base: str, asm_base: str, asm_funcs_base: str, src_path: str, arch: Optional[str]) -> Tuple[str, str, str]:
    # 基于项目已有的 map_input_to_output 映射 .o、.s 与 .json
    # 对交叉架构，加前缀子目录
    obj_root = obj_base if not arch or arch in {platform.machine()} else os.path.join(obj_base, arch)
    asm_root = asm_base if not arch or arch in {platform.machine()} else os.path.join(asm_base, arch)
    asm_funcs_root = asm_funcs_base if not arch or arch in {platform.machine()} else os.path.join(asm_funcs_base, arch)

    obj_path = map_input_to_output(input_base, obj_root, src_path)
    asm_path = map_input_to_output(input_base, asm_root, src_path)
    asm_funcs_path = map_input_to_output(input_base, asm_funcs_root, src_path)
    obj_path = os.path.splitext(obj_path)[0] + ".o"
    asm_path = os.path.splitext(asm_path)[0] + ".s"
    asm_funcs_path = os.path.splitext(asm_funcs_path)[0] + ".json"
    return obj_path, asm_path, asm_funcs_path


def discover_include_paths(src_path: str, project_root: str) -> List[str]:
    """启发式发现包含路径"""
    includes = []
    src_dir = os.path.dirname(src_path)
    
    # 常见包含目录模式
    common_patterns = [
        "include", "inc", "src", "lib", "common", "public", "headers",
        "third_party", "external", "vendor", "deps", "dependencies"
    ]
    
    # 从源码文件向上遍历，寻找包含目录
    current = src_dir
    while current and current != project_root:
        for pattern in common_patterns:
            inc_dir = os.path.join(current, pattern)
            if os.path.isdir(inc_dir):
                includes.append(f"-I{inc_dir}")
        current = os.path.dirname(current)
    
    # 添加项目根目录的常见包含路径
    for pattern in common_patterns:
        inc_dir = os.path.join(project_root, pattern)
        if os.path.isdir(inc_dir):
            includes.append(f"-I{inc_dir}")
    
    return includes


def discover_macros(src_path: str, project_root: str) -> List[str]:
    """启发式发现宏定义"""
    macros = []
    
    # 基于文件路径的宏猜测
    if "mbedtls" in src_path.lower():
        macros.extend(["-DMBEDTLS_CONFIG_FILE='mbedtls/mbedtls_config.h'"])
    if "openthread" in src_path.lower():
        macros.extend(["-DOPENTHREAD_CONFIG_FILE='openthread-config.h'"])
    if "curl" in src_path.lower():
        macros.extend(["-DCURL_STATICLIB"])
    
    # 通用宏
    macros.extend([
        "-D_GNU_SOURCE",
        "-D_POSIX_C_SOURCE=200809L",
        "-D_DEFAULT_SOURCE",
        "-DNDEBUG",  # 禁用调试断言
    ])
    
    return macros


def compile_to_object(chain: Toolchain, src_path: str, obj_path: str, project_root: str) -> bool:
    ensure_parent_dir(obj_path)
    
    # 确定编译器
    ext = os.path.splitext(src_path)[1].lower()
    if ext in {".cc", ".cpp", ".cxx", ".C"}:
        compiler = chain.gxx
        base_flags = chain.cxxflags
    else:
        compiler = chain.gcc
        base_flags = chain.cflags
    
    # 发现包含路径和宏
    includes = discover_include_paths(src_path, project_root)
    macros = discover_macros(src_path, project_root)
    
    # 构建完整命令
    cmd = [compiler, *base_flags, *includes, *macros, "-o", obj_path, src_path]
    code, out, err = run_cmd(cmd)
    return code == 0


def disassemble_object(chain: Toolchain, obj_path: str, asm_path: str) -> bool:
    ensure_parent_dir(asm_path)
    cmd = [chain.objdump, *chain.objdump_flags, obj_path]
    code, out, err = run_cmd(cmd)
    if code != 0:
        return False
    with open(asm_path, "w", encoding="utf-8") as f:
        f.write(out)
    return True


def parse_functions_from_objdump(asm_text: str) -> List[Dict[str, Any]]:
    """解析 GNU objdump -d 输出的函数。

    格式（示例）：
    0000000000000000 <function_name>:
       0:   55                      push   rbp
       ...
    下一段地址/空行/label 之前为该函数体。
    """
    functions: List[Dict[str, Any]] = []
    lines = asm_text.splitlines()

    def is_func_header(line: str) -> Optional[Tuple[str, str]]:
        # 形如: 0000000000000000 <name>:
        line = line.strip()
        if not line.endswith(":"):
            return None
        if "<" in line and ">:" in line:
            try:
                addr_part, rest = line.split(" ", 1)
            except ValueError:
                return None
            name = line[line.find("<") + 1: line.rfind(">")]
            return addr_part, name
        return None

    current: Dict[str, Any] = {}
    body_lines: List[str] = []
    func_id = 1

    for line in lines:
        header = is_func_header(line)
        if header:
            # 结束上一个函数
            if current:
                current["body"] = "\n".join(body_lines).rstrip()
                current["full_definition"] = current["body"]
                functions.append(current)
                current = {}
                body_lines = []

            addr, name = header
            current = {
                "function_id": func_id,
                "name": name,
                "start_addr": addr,
                "signature": name,  # 与源码结构保持相近字段
            }
            func_id += 1
        else:
            if current:
                body_lines.append(line)

    # 收尾
    if current:
        current["body"] = "\n".join(body_lines).rstrip()
        current["full_definition"] = current["body"]
        functions.append(current)

    # 估算大小（基于地址字段若可解析）——可选
    # 略：objdump 不总给 size，这里先跳过 end/size 计算
    return functions


def create_json_output(functions: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {
            **meta,
            "total_functions": len(functions),
            "extractor": "objdump function extractor",
        },
        "functions": functions,
    }


def process_all(if_complier: bool=False, if_disasm: bool=False) -> None:
    logger = setup_logging(PROJECT_ROOT)

    input_candidates = [
        os.path.join(PROJECT_ROOT, "resources", "datasets", "threelib"),
        os.path.join(PROJECT_ROOT, "src", "resources", "datasets", "threelib"),
    ]
    input_base = None
    for cand in input_candidates:
        if os.path.isdir(cand):
            input_base = cand
            break
    if input_base is None:
        raise FileNotFoundError("未找到输入目录 threelib")

    obj_base = os.path.join(PROJECT_ROOT, "resources", "datasets", "obj")
    asm_base = os.path.join(PROJECT_ROOT, "resources", "datasets", "asm")
    asm_funcs_base = os.path.join(PROJECT_ROOT, "resources", "datasets", "asm_funcs")
    os.makedirs(obj_base, exist_ok=True)
    os.makedirs(asm_base, exist_ok=True)
    os.makedirs(asm_funcs_base, exist_ok=True)

    chains = detect_toolchains()
    if not chains:
        logger.error("未发现可用编译/反汇编工具链。请安装 gcc/objdump 或交叉工具链。")
        return

    total_files = 0
    total_funcs = 0

    for root_dir, _, files in os.walk(input_base):
        # 跳过构建、测试、示例、工具等目录
        normalized = root_dir.replace("\\", "/")
        skip_patterns = [
            "/build/", "/.git/", "/test/", "/tests/", "/example/", "/examples/",
            "/tool/", "/tools/", "/demo/", "/demos/", "/sample/", "/samples/",
            "/benchmark/", "/benchmarks/", "/fuzz/", "/fuzzing/"
        ]
        if any(pattern in normalized for pattern in skip_patterns):
            continue
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            src_path = os.path.join(root_dir, name)

            for chain in chains:
                # 本机架构输出不加前缀，交叉架构加前缀子目录
                arch_prefix = None
                if chain.arch not in {platform.machine()}:
                    arch_prefix = chain.arch

                obj_path, asm_path, asm_funcs_path = map_src_to_obj_and_asm(
                    input_base, obj_base, asm_base, asm_funcs_base, src_path, arch_prefix
                )

                if if_complier:
                    # 编译
                    ok_compile = compile_to_object(chain, src_path, obj_path, PROJECT_ROOT)
                    if not ok_compile:
                        logger.error(f"编译失败({chain.arch}): {src_path}")
                        continue
                if if_disasm:
                    # 反汇编
                    ok_disas = disassemble_object(chain, obj_path, asm_path)
                    if not ok_disas:
                        logger.error(f"反汇编失败({chain.arch}): {obj_path}")
                        continue

                # 解析函数
                try:
                    with open(asm_path, "r", encoding="utf-8", errors="ignore") as f:
                        asm_text = f.read()
                    functions = parse_functions_from_objdump(asm_text)
                    if functions:
                        meta = {
                            "arch": chain.arch,
                            "source_file": os.path.relpath(src_path, PROJECT_ROOT),
                            "object_file": os.path.relpath(obj_path, PROJECT_ROOT),
                            "asm_file": os.path.relpath(asm_path, PROJECT_ROOT),
                            "extraction_timestamp": datetime.now().isoformat(),
                        }
                        json_out = create_json_output(functions, meta)
                        ensure_parent_dir(asm_funcs_path)
                        with open(asm_funcs_path, "w", encoding="utf-8") as jf:
                            json.dump(json_out, jf, ensure_ascii=False, indent=2)
                        total_files += 1
                        total_funcs += len(functions)
                        logger.info(
                            f"✅ {os.path.relpath(src_path, input_base)} -> {os.path.relpath(obj_path, obj_base)} | {os.path.relpath(asm_path, asm_base)} | {os.path.relpath(asm_funcs_path, asm_funcs_base)} (函数{len(functions)})"
                        )
                    else:
                        logger.info(
                            f"⏭️  {os.path.relpath(src_path, input_base)} ({chain.arch}) 无可解析函数"
                        )
                except Exception as e:
                    logger.error(f"解析失败({chain.arch}): {asm_path} | 错误: {e}")

    logger.info(f"完成。共输出 {total_files} 个文件，累计提取函数 {total_funcs} 个。")


def main():
    process_all()


if __name__ == "__main__":
    main()


