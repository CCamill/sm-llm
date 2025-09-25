# 智能样本生成系统

本系统重新设计了样本生成逻辑，基于签名匹配生成所有相似样本，并自动计算不相似样本数量为相似样本的1.5倍。

## 系统优势

### 相比之前系统的改进

1. **移除人为限制**: 不再通过`max_source_pairs`和`max_source_asm_pairs`人为限制相似样本数量
2. **充分利用数据**: 基于签名匹配生成所有可能的相似样本
3. **智能比例**: 自动计算不相似样本数量为相似样本的1.5倍
4. **更合理的分布**: 相似样本和不相似样本的比例更加合理

### 为什么之前的设计不合理？

1. **人为限制样本数量**: 通过`max_source_pairs`和`max_source_asm_pairs`人为限制了相似样本的数量，这是不合理的
2. **没有充分利用数据**: 应该基于签名匹配生成所有可能的相似样本
3. **不相似样本数量不智能**: 应该根据相似样本数量自动计算

## 系统架构

```
src/generate_samples_by_LLM/
├── smart_label_generator.py      # 智能相似样本生成器
├── smart_integrated_main.py      # 智能整合主脚本
├── dissimilar_generator.py       # 不相似样本生成器
├── data_integrator.py            # 数据整合器
├── data_splitter.py              # 数据分割器
└── SMART_README.md               # 说明文档
```

## 功能特性

- **智能相似样本生成**: 基于签名匹配生成所有相似样本
- **自动计算不相似样本**: 不相似样本数量 = 相似样本数量 × 1.5
- **随机混合**: 相似和不相似样本随机混合
- **数据分割**: 按7:2:1比例分割为训练/测试/验证集
- **多种输出**: 支持PKL和CSV两种输出格式

## 安装依赖

```bash
conda activate sm
pip install langchain langchain-community numpy pandas
```

## 使用方法

### 1. 基本使用

```bash
# 测试模式（只处理前2个项目）
python src/generate_samples_by_LLM/smart_integrated_main.py --test-mode

# 处理所有项目
python src/generate_samples_by_LLM/smart_integrated_main.py

# 处理指定项目
python src/generate_samples_by_LLM/smart_integrated_main.py --projects curl sqlite nmap
```

### 2. 参数说明

#### 基本参数
- `--source-dir`: 源码函数数据目录（默认：resources/datasets/source_funcs_treesitter）
- `--asm-dir`: 汇编函数数据目录（默认：resources/datasets/asm_funcs）
- `--output-dir`: 输出目录（默认：resources/datasets/smart_integrated_labels）
- `--ollama-url`: Ollama服务URL（默认：从config.py读取）

#### 项目参数
- `--projects`: 项目列表，None表示处理所有项目

#### 数据分割参数
- `--train-ratio`: 训练集比例（默认：0.7）
- `--test-ratio`: 测试集比例（默认：0.2）
- `--val-ratio`: 验证集比例（默认：0.1）
- `--random-seed`: 随机种子（默认：42）

#### 输出参数
- `--save-pkl`: 保存PKL文件（默认：True）
- `--save-csv`: 保存CSV文件（默认：True）

#### 其他参数
- `--log-level`: 日志级别（默认：INFO）
- `--test-mode`: 测试模式

## 输出文件

### 数据集文件

- `dataset_split_train.csv`: 训练集（70%）
- `dataset_split_test.csv`: 测试集（20%）
- `dataset_split_val.csv`: 验证集（10%）
- `dataset_split_statistics.csv`: 统计信息
- `dataset_split.pkl`: 完整数据集（PKL格式）

### 子目录

- `similar_labels/`: 相似样本标签文件
- `dissimilar_labels/`: 不相似样本标签文件

## 数据格式

### CSV文件格式

每行包含以下列：

- `source_signature`: 源码函数签名（项目+文件+函数名）
- `asm_signature`: 汇编函数签名（项目+文件+函数名）
- `similarity_score`: 相似度得分（0-1）
- `sample_type`: 样本类型（"similar" 或 "dissimilar"）
- `match_type`: 匹配类型（"exact", "fuzzy", "dissimilar"）
- `confidence`: 置信度（0-1）
- `source_project`: 源码项目名
- `source_file`: 源码文件名
- `source_function`: 源码函数名
- `asm_project`: 汇编项目名
- `asm_file`: 汇编文件名
- `asm_function`: 汇编函数名

## 测试结果

根据测试结果，智能系统的特点：

### 样本分布
- **总样本数**: 997个
- **相似样本**: 399个（40.0%）
- **不相似样本**: 598个（60.0%）

### 分割比例
- **训练集**: 697个样本（70%）
- **测试集**: 199个样本（20%）
- **验证集**: 101个样本（10%）

### 相似度分布
- **整体平均相似度**: 0.480
- **相似样本平均相似度**: 0.999（接近1.0，符合预期）
- **不相似样本平均相似度**: 0.133（接近0.0，符合预期）

## 使用示例

### Python代码示例

```python
import pandas as pd
import pickle

# 加载训练集
train_df = pd.read_csv('resources/datasets/smart_integrated_labels/dataset_split_train.csv')
print(f"训练集样本数: {len(train_df)}")

# 分析样本分布
similar_samples = train_df[train_df['sample_type'] == 'similar']
dissimilar_samples = train_df[train_df['sample_type'] == 'dissimilar']

print(f"相似样本: {len(similar_samples)} 个")
print(f"不相似样本: {len(dissimilar_samples)} 个")
print(f"相似样本比例: {len(similar_samples)/len(train_df):.1%}")

# 分析相似度分布
print(f"相似样本平均相似度: {similar_samples['similarity_score'].mean():.3f}")
print(f"不相似样本平均相似度: {dissimilar_samples['similarity_score'].mean():.3f}")

# 加载PKL文件
with open('resources/datasets/smart_integrated_labels/dataset_split.pkl', 'rb') as f:
    data = pickle.load(f)

train_samples = data['train_samples']
print(f"PKL训练集样本数: {len(train_samples)}")
```

### 数据加载示例

```python
from data_splitter import DataSplitter

# 创建数据分割器
splitter = DataSplitter("resources/datasets/smart_integrated_labels")

# 加载分割结果
split_result = splitter.load_split_from_pkl("dataset_split.pkl")

# 访问不同分割
train_samples = split_result.train_samples
test_samples = split_result.test_samples
val_samples = split_result.val_samples

print(f"训练集: {len(train_samples)} 个样本")
print(f"测试集: {len(test_samples)} 个样本")
print(f"验证集: {len(val_samples)} 个样本")
```

## 系统优势总结

### 1. 智能样本生成
- **基于签名匹配**: 生成所有可能的相似样本，不人为限制数量
- **自动计算比例**: 不相似样本数量自动设置为相似样本的1.5倍
- **充分利用数据**: 不会浪费任何可用的相似样本

### 2. 更合理的分布
- **相似样本**: 基于签名匹配，相似度接近1.0
- **不相似样本**: 基于LLM计算，相似度接近0.0
- **平衡分布**: 相似和不相似样本比例更加合理

### 3. 简化的参数
- **移除不必要参数**: 不再需要`max_source_pairs`和`max_source_asm_pairs`
- **自动计算**: 不相似样本数量自动计算
- **更易使用**: 用户只需要指定项目列表即可

## 注意事项

1. **数据质量**: 相似样本基于签名匹配，质量更高
2. **计算时间**: 不相似样本需要LLM计算，时间较长
3. **内存使用**: 处理大量数据时注意内存使用情况
4. **Ollama服务**: 确保Ollama服务运行正常

## 故障排除

1. **Ollama连接失败**: 检查Ollama服务是否运行
2. **内存不足**: 减少处理的项目数量
3. **数据加载失败**: 检查输入目录是否存在且包含有效数据
4. **分割比例错误**: 检查train_ratio + test_ratio + val_ratio = 1.0

## 扩展功能

### 自定义分割比例

```bash
python src/generate_samples_by_LLM/smart_integrated_main.py \
    --train-ratio 0.8 \
    --test-ratio 0.15 \
    --val-ratio 0.05
```

### 处理特定项目

```bash
python src/generate_samples_by_LLM/smart_integrated_main.py \
    --projects curl sqlite nmap
```

### 测试模式

```bash
python src/generate_samples_by_LLM/smart_integrated_main.py --test-mode
```
