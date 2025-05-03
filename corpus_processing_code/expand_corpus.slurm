#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=expand_corpus
#SBATCH --mail-user=kf2365@nyu.edu
#SBATCH --mail-type=END,FAIL  # 任务结束或失败时邮件通知
#SBATCH --output=logs/experiment_%j.out  # %j 代表 Job ID
#SBATCH --error=logs/experiment_%j.err  # 记录错误日志

# 创建 logs 目录
mkdir -p logs

# Singularity 镜像路径
EXT3_PATH="/scratch/kf2365/numsport/overlay-50G-10M.ext3:ro"
SIF_PATH="/scratch/kf2365/numsport/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Python 脚本路径
SCRIPT_PATH="numsports/corpus_processing_code/expand_corpus_form.py"

# 激活 Singularity 容器
singularity exec --nv \
--overlay ${EXT3_PATH}:ro \
${SIF_PATH} /bin/bash -c "
    source /ext3/env.sh  
    python3 ${SCRIPT_PATH}
    "

