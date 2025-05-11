#!/bin/bash

# python scripts/run_direct_gen.py \
#     --dataset_name num_sports \
#     --split test \
#     --model_path Qwen/QwQ-32B-Preview


python scripts/run_naive_rag.py \
    --dataset_name num_sports \
    --split test \
    --use_jina True \
    --model_path Qwen/QwQ-32B-Preview \
    --jina_api_key jina_492b792f8c3d4dca962b9b2c8b7e039by_76MCmDFlBZcZTWaYjP6EqEKbXA \
    --bing_endpoint https://serpapi.com/search?engine=bing \
    --bing_subscription_key 28e18c6bedf837497cca680c3a608bf97d145412c9f508a465d44b6a53038e1f