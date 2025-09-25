"""测试智能样本生成系统"""

import os
import sys
import logging

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import ollama_url
from smart_label_generator import SmartLabelGenerator
from dissimilar_generator import DissimilarSampleGenerator
from data_integrator import DataIntegrator
from data_splitter import DataSplitter

def test_smart_system():
    """测试智能样本生成系统"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("开始测试智能样本生成系统...")
    
    try:
        # 1. 测试智能相似样本生成器
        logger.info("测试智能相似样本生成器...")
        similar_generator = SmartLabelGenerator(
            source_dir="resources/datasets/source_funcs_treesitter",
            asm_dir="resources/datasets/asm_funcs",
            output_dir="resources/datasets/test_smart_labels",
            ollama_url=ollama_url
        )
        
        # 测试单个项目
        result = similar_generator.generate_all_similar_samples("curl")
        logger.info(f"curl项目生成了 {result.total_samples} 个相似样本")
        
        # 2. 测试不相似样本生成器
        logger.info("测试不相似样本生成器...")
        dissimilar_generator = DissimilarSampleGenerator(
            source_dir="resources/datasets/source_funcs_treesitter",
            asm_dir="resources/datasets/asm_funcs",
            output_dir="resources/datasets/test_smart_labels",
            ollama_url=ollama_url
        )
        
        # 生成少量不相似样本
        saved_files = dissimilar_generator.generate_and_save(
            max_pairs=10,
            min_different_projects=1,
            batch_size=5,
            delay=0.2,
            save_pkl=True,
            save_csv=True
        )
        
        if 'pkl' in saved_files:
            dissimilar_samples = dissimilar_generator.load_from_pkl(saved_files['pkl'])
            logger.info(f"生成了 {len(dissimilar_samples)} 个不相似样本")
        
        # 3. 测试数据整合器
        logger.info("测试数据整合器...")
        integrator = DataIntegrator("resources/datasets/test_smart_labels")
        
        # 直接使用生成的样本，而不是从文件加载
        similar_samples = []
        for sample in result.samples:
            from data_loader import FunctionData
            from data_integrator import IntegratedSample
            
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
        
        logger.info(f"准备了 {len(similar_samples)} 个相似样本")
        
        # 转换不相似样本
        dissimilar_samples = []
        for pair in dissimilar_samples:
            from data_integrator import IntegratedSample
            
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
            dissimilar_samples.append(integrated_sample)
        
        logger.info(f"准备了 {len(dissimilar_samples)} 个不相似样本")
        
        # 整合样本
        all_samples = integrator.integrate_samples(
            similar_samples=similar_samples,
            dissimilar_samples=dissimilar_samples,
            shuffle=True
        )
        logger.info(f"整合后共有 {len(all_samples)} 个样本")
        
        # 4. 测试数据分割器
        logger.info("测试数据分割器...")
        splitter = DataSplitter("resources/datasets/test_smart_labels")
        
        # 分割数据集
        split_result = splitter.split_dataset(
            samples=all_samples,
            train_ratio=0.7,
            test_ratio=0.2,
            val_ratio=0.1,
            shuffle=True,
            random_seed=42
        )
        
        logger.info(f"分割结果:")
        logger.info(f"  训练集: {len(split_result.train_samples)} 个样本")
        logger.info(f"  测试集: {len(split_result.test_samples)} 个样本")
        logger.info(f"  验证集: {len(split_result.val_samples)} 个样本")
        
        # 验证分割结果
        if splitter.validate_split(split_result):
            logger.info("分割结果验证通过")
        else:
            logger.warning("分割结果验证失败")
        
        # 5. 保存结果
        saved_files = splitter.save_split_to_csv(split_result)
        logger.info(f"保存了 {len(saved_files)} 个文件")
        
        logger.info("智能样本生成系统测试完成!")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_smart_system()
    if success:
        print("测试成功!")
    else:
        print("测试失败!")
        sys.exit(1)
