#!/bin/bash
# run_distributed.sh - 在主节点上运行此脚本启动分布式推理

set -e

# ============== 配置区域 ==============

# 模型名称
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# hostfile 路径
HOSTFILE="hostfile"

# DeepSpeed 配置
DS_CONFIG="ds_config.json"

# 主节点 IP（运行此脚本的机器）
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-29500}"

# GPU 数量（每个节点）
GPUS_PER_NODE=1

# 总节点数
NUM_NODES=2

# =====================================

echo "=========================================="
echo "启动分布式推理"
echo "=========================================="
echo "模型: $MODEL_NAME"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"
echo "节点数: $NUM_NODES"
echo "每节点 GPU: $GPUS_PER_NODE"
echo "=========================================="

# 检查 hostfile
if [ ! -f "$HOSTFILE" ]; then
    echo "错误: hostfile 不存在: $HOSTFILE"
    echo "请先配置 hostfile"
    exit 1
fi

echo "Hostfile 内容:"
cat "$HOSTFILE"
echo ""

# 使用 DeepSpeed launcher 启动
deepspeed \
    --hostfile="$HOSTFILE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --no_local_rank \
    run_inference.py \
    --model "$MODEL_NAME" \
    --config "$DS_CONFIG"

echo ""
echo "=========================================="
echo "分布式推理完成"
echo "=========================================="
