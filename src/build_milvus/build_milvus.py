import os
import json
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple


class FunctionSearchSystem:
    """基于Milvus的函数源码存储和检索系统"""
    
    def __init__(self, host="localhost", port="19530", collection_name="function_codes"):
        """
        初始化系统
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
        """
        self.collection_name = collection_name
        self.embedding_dim = 768  # sentence-transformers模型的默认维度
        
        # 连接Milvus
        connections.connect(host=host, port=port)
        print(f"成功连接到 Milvus: {host}:{port}")
        
        # 加载嵌入模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("嵌入模型加载完成")
        
        # 创建或加载集合
        self._create_collection()
        
    def _create_collection(self):
        """创建Milvus集合（如果不存在）"""
        if utility.has_collection(self.collection_name):
            print(f"集合 '{self.collection_name}' 已存在，直接加载")
            self.collection = Collection(self.collection_name)
        else:
            # 定义字段schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="project_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="function_code", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            # 创建schema
            schema = CollectionSchema(fields=fields, description="函数源码存储集合")
            
            # 创建集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"成功创建集合 '{self.collection_name}'")
            
            # 创建索引以加速检索
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="/", index_params=index_params)
            print("索引创建完成")
    
    def load_functions_from_directory(self, root_dir: str):
        """
        从三级目录结构加载函数数据
        
        目录结构：
        root_dir/
            project1/
                file1.json
                file2.json
            project2/
                file3.json
        
        Args:
            root_dir: 根目录路径
        """
        all_data = []
        
        # 遍历一级目录（项目名称）
        for project_name in os.listdir(root_dir):
            project_path = os.path.join(root_dir, project_name)
            
            if not os.path.isdir(project_path):
                continue
            
            print(f"正在处理项目: {project_name}")
            
            # 遍历二级目录（JSON文件）
            for json_file in os.listdir(project_path):
                if not json_file.endswith('.json'):
                    continue
                
                json_path = os.path.join(project_path, json_file)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 提取functions字段
                    functions = data.get('functions', [])
                    
                    for func in functions:
                        function_name = func.get('function_name', '')
                        function_code = func.get('full_definition', '')
                        
                        if function_name and function_code:
                            all_data.append({
                                'project_name': project_name,
                                'file_name': json_file,
                                'function_name': function_name,
                                'function_code': function_code
                            })
                    
                    print(f"  - 文件 {json_file}: 加载 {len(functions)} 个函数")
                
                except Exception as e:
                    print(f"  - 错误: 无法读取文件 {json_path}: {e}")
        
        print(f"\n总共加载 {len(all_data)} 个函数")
        
        if all_data:
            self._insert_functions(all_data)
        
        return len(all_data)
    
    def _insert_functions(self, functions: List[Dict]):
        """
        将函数数据插入Milvus
        
        Args:
            functions: 函数数据列表
        """
        print("\n开始生成向量嵌入...")
        
        # 生成嵌入向量（结合函数名和函数代码）
        texts = [f"{func['function_name']}\n{func['function_code']}" for func in functions]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 准备插入数据
        insert_data = [
            [func['project_name'] for func in functions],
            [func['file_name'] for func in functions],
            [func['function_name'] for func in functions],
            [func['function_code'] for func in functions],
            embeddings.tolist()
        ]
        
        # 插入数据
        print("正在插入数据到Milvus...")
        self.collection.insert(insert_data)
        self.collection.flush()
        
        print(f"成功插入 {len(functions)} 条数据")
    
    def search_functions(self, query: str, k: int = 5) -> List[Dict]:
        """
        根据查询检索语义最相近的k个函数
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            检索结果列表，每个结果包含函数信息和相似度分数
        """
        # 加载集合到内存
        self.collection.load()
        
        # 生成查询向量
        query_embedding = self.model.encode([query])[0].tolist()
        
        # 设置搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["project_name", "file_name", "function_name", "function_code"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'project_name': hit.entity.get('project_name'),
                    'file_name': hit.entity.get('file_name'),
                    'function_name': hit.entity.get('function_name'),
                    'function_code': hit.entity.get('function_code'),
                    'distance': hit.distance,  # L2距离，越小越相似
                    'similarity_score': 1 / (1 + hit.distance)  # 转换为相似度分数
                })
        
        return formatted_results
    
    def delete_collection(self):
        """删除集合"""
        utility.drop_collection(self.collection_name)
        print(f"集合 '{self.collection_name}' 已删除")
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        self.collection.load()
        num_entities = self.collection.num_entities
        return {
            'collection_name': self.collection_name,
            'num_functions': num_entities
        }


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    system = FunctionSearchSystem(
        host="localhost",
        port="19530",
        collection_name="function_codes"
    )
    
    # 1. 加载函数数据（假设数据在 ./data 目录下）
    data_dir = "resources/datasets/source_funcs_treesitter"
    num_loaded = system.load_functions_from_directory(data_dir)
    print(f"\n总共加载了 {num_loaded} 个函数")
    
    # 2. 查看集合统计信息
    stats = system.get_collection_stats()
    print(f"\n集合统计: {stats}")
    
    # 3. 执行检索
    query = "static void\npanic( const char*  fmt,\n       ... )\n{\n  va_list  ap;\n\n\n  fprintf( stderr, \"PANIC: \" );\n\n  va_start( ap, fmt );\n  vfprintf( stderr, fmt, ap );\n  va_end( ap );\n\n  fprintf( stderr, \"\\n\" );\n\n  exit(2);\n}"
    results = system.search_functions(query, k=5)
    
    print(f"\n查询: '{query}'")
    print(f"找到 {len(results)} 个相关函数:\n")
    
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"  项目: {result['project_name']}")
        print(f"  文件: {result['file_name']}")
        print(f"  函数名: {result['function_name']}")
        print(f"  相似度分数: {result['similarity_score']:.4f}")
        print(f"  代码预览: {result['function_code'][:100]}...")
        print()