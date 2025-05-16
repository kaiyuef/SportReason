#!/bin/bash

python baselines/Search-o1/scripts/run_search_o1_numsport.py \
  --data_file dataset/num_sports_500.jsonl \
  --subset_num 20 \
  --max_search_limit 5 \
  --max_turn 10 \
  --top_k 10 \
  --max_doc_len 3000 \
  --temperature 0.3 \
  --top_p 0.9 \
  --bing_subscription_key "392c6ae22af04ea6b19883ef493162ed9d3b660c81186b4d153cfc514b7f918d" \
  --bing_endpoint https://serpapi.com/search?engine=bing \
  --openai_base_url "https://api.together.xyz/v1" \
  --openai_api_key e4c956ab9bca2dd990f6b27369a471065664731dacffbd3f5d07264da9aa3c62 \
  --openai_model Qwen/QwQ-32B 