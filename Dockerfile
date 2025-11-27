# Dockerfile
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# 设置工作目录
WORKDIR /workspace

# 安装DeepSpeed和其他依赖
RUN pip install -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV NCCL_DEBUG=INFO

# 创建非root用户（可选，但推荐）
RUN useradd -m -u 1000 deepspeed-user
USER deepspeed-user