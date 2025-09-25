"""智能整合主脚本，自动生成所有相似样本并智能计算不相似样本数量"""

import os
import sys
import argparse
import logging
from typing import List, Optional, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import ollama_url
from smart_label_generator import SmartLabelGenerator
from dissimilar_generator import DissimilarSampleGenerator
from data_integrator import DataIntegrator, IntegratedSample
from data_splitter import DataSplitter, DatasetSplit
from data_loader import FunctionData

def setup_logging(log_level: str = "INFO") -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('smart_integrated_generation.log', encoding='utf-8')
        ]
    )

def generate_all_similar_samples(source_dir: str, 
                               asm_dir: str, 
                               output_dir: str,
                               projects: List[str],
                               ollama_url: str) -> List[IntegratedSample]:
    """
    生成所有相似样本
    
    Args:
        source_dir: 源码函数数据目录
        asm_dir: 汇编函数数据目录
        output_dir: 输出目录
        projects: 项目列表
        ollama_url: Ollama服务URL
        
    Returns:
        List[IntegratedSample]: 相似样本列表
    """
    logger = logging.getLogger(__name__)
    logger.info("开始生成所有相似样本...")
    
    # 创建智能相似样本生成器
    similar_generator = SmartLabelGenerator(
        source_dir=source_dir,
        asm_dir=asm_dir,
        output_dir=os.path.join(output_dir, "similar_labels"),
        ollama_url=ollama_url
    )
    
    # 生成所有相似样本
    similar_results = similar_generator.generate_all_projects_similar_samples(projects)
    
    # 保存所有相似样本到CSV
    similar_generator.save_all_results_to_csv(similar_results)
    
    # 转换为IntegratedSample格式
    similar_samples = []
    total_similar_count = 0
    
    for result in similar_results:
        total_similar_count += result.total_samples
        for sample in result.samples:
            # 创建FunctionData对象
            source_func = FunctionData(
                function_id=sample['source_function']['function_id'],
                function_name=sample['source_function']['function_name'],
                signature=sample['source_function']['signature'],
                body=sample['source_function']['body'],
                full_definition=sample['source_function']['full_definition'],
                source_file=sample['source_function']['source_file'],
                project_name=sample['source_function']['file_name'].split('/')[0] if '/' in sample['source_function']['file_name'] else 'unknown',
                file_name=sample['source_function']['file_name']
            )
            
            asm_func = FunctionData(
                function_id=sample['asm_function']['function_id'],
                function_name=sample['asm_function']['function_name'],
                signature=sample['asm_function']['signature'],
                body=sample['asm_function']['body'],
                full_definition=sample['asm_function']['full_definition'],
                source_file=sample['asm_function']['source_file'],
                project_name=sample['asm_function']['file_name'].split('/')[0] if '/' in sample['asm_function']['file_name'] else 'unknown',
                file_name=sample['asm_function']['file_name']
            )
            
            # 创建整合样本
            integrated_sample = IntegratedSample(
                source_func=source_func,
                asm_func=asm_func,
                similarity_score=sample["similarity_label"],
                sample_type="similar",
                match_type=sample["match_type"],
                confidence=sample["confidence"],
                source_signature=f"{source_func.project_name}+{source_func.file_name}+{source_func.function_name}",
                asm_signature=f"{asm_func.project_name}+{asm_func.file_name}+{asm_func.function_name}",
                metadata=sample["metadata"]
            )
            
            similar_samples.append(integrated_sample)
    
    logger.info(f"生成了 {len(similar_samples)} 个相似样本（来自 {len(similar_results)} 个项目）")
    return similar_samples

def generate_dissimilar_samples_auto(source_dir: str,
                                   asm_dir: str,
                                   output_dir: str,
                                   similar_count: int,
                                   ollama_url: str) -> List[IntegratedSample]:
    """
    自动生成不相似样本，数量为相似样本的1.5倍
    
    Args:
        source_dir: 源码函数数据目录
        asm_dir: 汇编函数数据目录
        output_dir: 输出目录
        similar_count: 相似样本数量
        ollama_url: Ollama服务URL
        
    Returns:
        List[IntegratedSample]: 不相似样本列表
    """
    logger = logging.getLogger(__name__)
    
    # 自动计算不相似样本数量
    dissimilar_count = int(similar_count * 1.5)
    logger.info(f"相似样本数量: {similar_count}, 自动设置不相似样本数量: {dissimilar_count}")
    
    # 创建不相似样本生成器
    dissimilar_generator = DissimilarSampleGenerator(
        source_dir=source_dir,
        asm_dir=asm_dir,
        output_dir=os.path.join(output_dir, "dissimilar_labels"),
        ollama_url=ollama_url
    )
    
    # 生成不相似样本
    saved_files = dissimilar_generator.generate_and_save(
        max_pairs=dissimilar_count,
        min_different_projects=1,
        batch_size=50,
        delay=0.1,
        save_pkl=True,
        save_csv=True
    )
    
    # 从PKL文件加载样本
    dissimilar_samples = []
    if 'pkl' in saved_files:
        dissimilar_samples = dissimilar_generator.load_from_pkl(saved_files['pkl'])
    
    # 转换为IntegratedSample格式
    integrated_dissimilar_samples = []
    for pair in dissimilar_samples:
        integrated_sample = IntegratedSample(
            source_func=pair.source_func,
            asm_func=pair.asm_func,
            similarity_score=pair.similarity_score,
            sample_type="dissimilar",
            match_type="dissimilar",
            confidence=0.8,
            source_signature=str(pair.source_signature),
            asm_signature=str(pair.asm_signature),
            metadata={
                "generation_method": "dissimilar_llm",
                "source_project": pair.source_signature.project_name,
                "source_file": pair.source_signature.file_name,
                "asm_project": pair.asm_signature.project_name,
                "asm_file": pair.asm_signature.file_name
            }
        )
        integrated_dissimilar_samples.append(integrated_sample)
    
    logger.info(f"生成了 {len(integrated_dissimilar_samples)} 个不相似样本")
    return integrated_dissimilar_samples

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能生成相似和不相似样本并分割数据集")
    
    # 基本参数
    parser.add_argument("--source-dir", 
                       default="resources/datasets/source_funcs_treesitter",
                       help="源码函数数据目录")
    parser.add_argument("--asm-dir", 
                       default="resources/datasets/asm_funcs",
                       help="汇编函数数据目录")
    parser.add_argument("--output-dir", 
                       default="resources/datasets/smart_integrated_labels",
                       help="输出目录")
    parser.add_argument("--ollama-url", 
                       default=ollama_url,
                       help="Ollama服务URL")
    
    # 项目参数
    parser.add_argument("--projects", 
                       nargs="+",
                       default=None,    # None表示处理所有项目
                       help="项目列表，None表示处理所有项目")
    
    # 数据分割参数
    parser.add_argument("--train-ratio", 
                       type=float, 
                       default=0.7,
                       help="训练集比例")
    parser.add_argument("--test-ratio", 
                       type=float, 
                       default=0.2,
                       help="测试集比例")
    parser.add_argument("--val-ratio", 
                       type=float, 
                       default=0.1,
                       help="验证集比例")
    parser.add_argument("--random-seed", 
                       type=int, 
                       default=42,
                       help="随机种子")
    
    # 输出参数
    parser.add_argument("--save-pkl", 
                       action="store_true",
                       default=True,
                       help="保存PKL文件")
    parser.add_argument("--save-csv", 
                       action="store_true",
                       default=True,
                       help="保存CSV文件")
    
    # 日志参数
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="日志级别")
    
    # 测试参数
    parser.add_argument("--test-mode", 
                       action="store_true",
                       help="测试模式，只处理少量数据")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 测试模式调整参数
    if args.test_mode:
        # 测试模式只处理前2个项目
        if args.projects is None:
            from data_loader import DataLoader
            data_loader = DataLoader(args.source_dir, args.asm_dir)
            all_projects = data_loader.get_all_projects()
            args.projects = all_projects[:2]
        logger.info("测试模式：只处理前2个项目")
    
    # 检查目录是否存在
    if not os.path.exists(args.source_dir):
        logger.error(f"源码目录不存在: {args.source_dir}")
        return 1
    
    if not os.path.exists(args.asm_dir):
        logger.error(f"汇编目录不存在: {args.asm_dir}")
        return 1
    
    try:
        # 1. 生成所有相似样本
        similar_samples = generate_all_similar_samples(
            source_dir=args.source_dir,
            asm_dir=args.asm_dir,
            output_dir=args.output_dir,
            projects=args.projects,
            ollama_url=args.ollama_url
        )
        
        if not similar_samples:
            logger.error("没有生成任何相似样本")
            return 1
        
        # 2. 自动生成不相似样本（数量为相似样本的1.5倍）
        dissimilar_samples = generate_dissimilar_samples_auto(
            source_dir=args.source_dir,
            asm_dir=args.asm_dir,
            output_dir=args.output_dir,
            similar_count=len(similar_samples),
            ollama_url=args.ollama_url
        )
        
        # 3. 整合样本
        logger.info("开始整合样本...")
        integrator = DataIntegrator(args.output_dir)
        all_samples = integrator.integrate_samples(
            similar_samples=similar_samples,
            dissimilar_samples=dissimilar_samples,
            shuffle=True
        )
        
        # 4. 分割数据集
        logger.info("开始分割数据集...")
        splitter = DataSplitter(args.output_dir)
        split_result = splitter.split_dataset(
            samples=all_samples,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            shuffle=True,
            random_seed=args.random_seed
        )
        
        # 5. 验证分割结果
        if not splitter.validate_split(split_result):
            logger.warning("分割结果验证失败，但继续处理")
        
        # 6. 保存结果
        saved_files = {}
        
        if args.save_csv:
            csv_files = splitter.save_split_to_csv(split_result)
            saved_files.update(csv_files)
        
        if args.save_pkl:
            pkl_file = splitter.save_split_to_pkl(split_result)
            saved_files["pkl"] = pkl_file
        
        # 7. 打印结果
        logger.info("=" * 60)
        logger.info("智能整合样本生成和数据集分割完成!")
        logger.info(f"相似样本: {len(similar_samples)} 个")
        logger.info(f"不相似样本: {len(dissimilar_samples)} 个")
        logger.info(f"总样本数: {len(all_samples)} 个")
        logger.info(f"训练集: {len(split_result.train_samples)} 个样本")
        logger.info(f"测试集: {len(split_result.test_samples)} 个样本")
        logger.info(f"验证集: {len(split_result.val_samples)} 个样本")
        logger.info(f"不相似样本比例: {len(dissimilar_samples)/len(all_samples):.1%}")
        logger.info("=" * 60)
        
        if saved_files:
            logger.info("生成的文件:")
            for file_type, file_path in saved_files.items():
                logger.info(f"  - {file_type.upper()}: {file_path}")
        else:
            logger.warning("没有生成任何文件")
        
        return 0
        
    except Exception as e:
        logger.error(f"智能整合样本生成失败: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
