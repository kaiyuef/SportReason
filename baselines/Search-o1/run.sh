#!/bin/bash

python baselines/Search-o1/scripts/run_search_o1_numsport.py \
  --dataset_name num_sports_500 \
  --data_file baselines/Search-o1/data/num_sports_500.jsonl \
  --subset_num 500 \
  --max_search_limit 10 \
  --max_turn 10 \
  --top_k 10 \
  --max_doc_len 4096 \
  --temperature 0.3 \
  --top_p 0.9 \
  --bing_subscription_key "392c6ae22af04ea6b19883ef493162ed9d3b660c81186b4d153cfc514b7f918d" \
  --bing_endpoint https://serpapi.com/search?engine=bing \
  --openai_base_url "https://generativelanguage.googleapis.com/v1beta/openai/" \
  --openai_api_key "AIzaSyBIxoRNoYPPgCj8SGC6b_wjArEzn1QTKLA" \
  --openai_model "gemini-2.5-flash-preview-04-17" \
  --use_jina \
  --jina_api_key "jina_492b792f8c3d4dca962b9b2c8b7e039by_76MCmDFlBZcZTWaYjP6EqEKbXA"