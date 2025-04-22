#bin/bash

python3 run_generation.py \
--model_name_or_path "/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
--bf16 \
--trim_logits \
--batch_size 1 \
--use_hpu_graphs \
--use_kv_cache  \
--max_new_tokens 500 \
--parallel_strategy "ep" \
--prompt "What should i choose a brand of Gaming gears Razer, Corsair or Steelseries?" \
