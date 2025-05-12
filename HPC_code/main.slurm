#!/bin/bash
#SBATCH --job-name=e5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --output=logs/qa_retrieval_%j_%N.out
#SBATCH --error=logs/qa_retrieval_%j_%N.err
#SBATCH --mail-user=kf2365@nyu.edu
#SBATCH --mail-type=END,FAIL

# ---------- 容器与 Overlay ----------
EXT3_PATH="/scratch/kf2365/numsport/overlay-50G-10M.ext3:ro"
SIF_PATH="/scratch/kf2365/numsport/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

# ---------- QA-pipeline 参数 ----------
INPUT_PATH="numsports/dataset/merged_dataset_final.jsonl"

EMBED_MODEL="intfloat/e5-mistral-7b-instruct"  # 模型名称
RERANKER_MODEL="none"
MODEL_SAFE=$(echo $EMBED_MODEL | tr '/' '_')
RERANKER_SAFE=$(echo $RERANKER_MODEL | tr '/' '_')
OUTPUT_DIR="numsports/answer_results/${MODEL_SAFE}/${RERANKER_SAFE}"

RETRIEVER_MODE="chroma"    # bm25 | chroma | hybrid
DISTANCE="cosine"        # cosine | l2

READER_MODEL="qwen"      # qwen | llama

# 三库 Top-K （送入重排的候选数）
TEXT_K=100
TABLE_K=25
INFOBOX_K=40

# 重排后保留数（对 retrieval-only 无影响）
FINAL_TEXT_K=12
FINAL_TABLE_K=6
FINAL_INFOBOX_K=10

# 重排显存开关（对 retrieval-only 无影响）
RERANKER_BATCH=96
MAX_RERANK_DOCS=170     # 0 or neg = 不裁剪

# slice 控制：只处理第 START_IDX .. END_IDX-1 题
START_IDX=0
END_IDX=3100

DEVICE="cuda:0"

# ---------- 日志目录 ----------
mkdir -p logs

# ---------- 执行 ----------
singularity exec --nv \
--overlay ${EXT3_PATH} \
${SIF_PATH} /bin/bash -c "
    source /ext3/env.sh
    python numsports/pipeline.py \
      --input \"${INPUT_PATH}\" \
      --output \"${OUTPUT_DIR}\" \
      --embed_model \"${EMBED_MODEL}\" \
      --retriever \"${RETRIEVER_MODE}\" \
      --distance \"${DISTANCE}\" \
      --reranker_model \"${RERANKER_MODEL}\" \
      --reader \"${READER_MODEL}\" \
      --device \"${DEVICE}\" \
      --text_k ${TEXT_K} \
      --table_k ${TABLE_K} \
      --infobox_k ${INFOBOX_K} \
      --final_text_k ${FINAL_TEXT_K} \
      --final_table_k ${FINAL_TABLE_K} \
      --final_infobox_k ${FINAL_INFOBOX_K} \
      --reranker_batch ${RERANKER_BATCH} \
      --max_rerank_docs ${MAX_RERANK_DOCS} \
      --start_idx ${START_IDX} \
      --end_idx   ${END_IDX} \
      --stage retrieval \
"
