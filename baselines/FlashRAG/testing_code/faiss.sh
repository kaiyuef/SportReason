#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=flashrag_index
#SBATCH --mail-user=kf2365@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------- 预处理 ----------------
mkdir -p logs


# Singularity 镜像 & Overlay
EXT3_PATH="/scratch/kf2365/numsport/overlay-50G-10M.ext3:ro"
SIF_PATH="/scratch/kf2365/numsport/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# ---------- 脚本与路径 ----------
CORPUS_PATH="numsports/flashrag/normalized_corpus.jsonl"           # 每行 {"id":..., "contents":...}
INDEX_SAVE_DIR="flashrag/indexes/wiki_faiss_bge"
MODEL_PATH="BAAI/bge-m3"

# ---------- 可调参数 ----------
FAISS_TYPE="Flat"       # 可选: Flat, IVF, IVF_PQ
RETRIEVAL_METHOD="bge"  # 可选: e5, gte, etc.

mkdir -p ${INDEX_SAVE_DIR}


singularity exec --nv \
  --overlay ${EXT3_PATH} \
  ${SIF_PATH} /bin/bash -c "
    source /ext3/env.sh
    python bFlashRAG/flashrag/retriever/index_builder.py \
      --corpus_path numsports/flashrag/normalized_corpus.jsonl \
      --model_name BAAI/bge-m3 \
      --model_path "BAAI/bge-m3"
      --output_dir flashrag/indexes/wiki_faiss_bge \
      --fp16 \
      --normalize \
      --batch_size 32 \
      --max_length 4096 \
      --faiss_type Flat \
      --save_embeddings
"
