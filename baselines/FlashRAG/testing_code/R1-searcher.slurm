#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2        
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=R1
#SBATCH --mail-user=kf2365@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ─────────── 预处理 ───────────
mkdir -p logs

# Singularity 镜像 & Overlay
EXT3_PATH="/scratch/kf2365/numsport/overlay-50G-10M.ext3:ro"
SIF_PATH="/scratch/kf2365/numsport/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"



# ─────────── 运行 naive rag ───────────
singularity exec --nv --overlay ${EXT3_PATH} ${SIF_PATH} /bin/bash -c "
  source /ext3/env.sh           # ↙ 若 overlay 内有 conda/venv 初始化脚本
  python FlashRAG/testing_code/R1-searcher.py \
"
