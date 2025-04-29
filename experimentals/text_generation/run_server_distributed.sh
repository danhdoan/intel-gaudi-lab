#!/bin/bash

# ==============================================================================

NUM_WORLD_SIZE=${1:-8}
export MODEL_PATH=${2:-"/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}

# ==============================================================================

python gaudi_spawn.py \
    --use_deepspeed \
    --world_size $NUM_WORLD_SIZE \
    server.py

# ==============================================================================
