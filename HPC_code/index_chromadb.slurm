#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=8:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=chromadb_batch_index
#SBATCH --mail-user=kf2365@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out   # %x=job-name  %j=job-id
#SBATCH --error=logs/%x_%j.err

# ---------------- 预处理 ----------------
mkdir -p logs

# Singularity 镜像 & Overlay
EXT3_PATH="/scratch/kf2365/numsport/overlay-50G-10M.ext3:ro"
SIF_PATH="/scratch/kf2365/numsport/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# Python 脚本（批量处理各子目录）
SCRIPT_PATH="numsports/corpus_processing_code/index_chromadb.py"

# ---------- 可自定义参数 ----------
# 留空则采用脚本默认值；如需修改直接填路径 / 名称
BASE_CORPUS_DIR="numsports/corpus"
BASE_PERSIST_DIR="numsports/indexes"
MODEL_NAME="BAAI/bge-m3"

ARGS="--base_corpus_dir ${BASE_CORPUS_DIR} \
      --base_persist_dir ${BASE_PERSIST_DIR} \
      --model_name ${MODEL_NAME}"

# ---------------- 运行 ----------------
singularity exec --nv \
  --overlay ${EXT3_PATH} \
  ${SIF_PATH} /bin/bash -c "
      source /ext3/env.sh
      python3 ${SCRIPT_PATH} ${ARGS}
"
