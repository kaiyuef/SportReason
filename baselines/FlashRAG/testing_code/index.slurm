#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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

# ---------- 脚本与路径 ----------         # 每行 {"id":..., "contents":...}
INDEX_SAVE_DIR="FlashRAG/indexes/wiki_faiss_bge_final"

mkdir -p ${INDEX_SAVE_DIR}

# ---------------- 运行构建 ----------------
singularity exec --nv \
  --overlay ${EXT3_PATH} \
  ${SIF_PATH} /bin/bash -c "
      source /ext3/env.sh
      python FlashRAG/flashrag/retriever/index_builder.py \
        --retrieval_method bge \
        --corpus_path FlashRAG/corpus/normalized_corpus.jsonl \
        --model_path .cache/hub/models/bge-m3 \
        --save_dir FlashRAG/indexes/wiki_faiss_bge_final \
        --batch_size 64 \
        --max_length 512 \
        --use_fp16 \
        --pooling_method mean \
        --sentence_transformer \
        --faiss_gpu \
        --faiss_type Flat \
        --save_embedding \
"

