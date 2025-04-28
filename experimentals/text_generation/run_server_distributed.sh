#bin/bash

python3 gaudi_spawn.py --use_deepspeed \
--world_size 8 server.py \
--model_name_or_path "/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
--bf16 \
--trim_logits \
--batch_size 1 \
--use_hpu_graphs \
--use_kv_cache  \
--max_new_tokens 500 \
--parallel_strategy "ep" \
--prompt "How to cook a pancake?"