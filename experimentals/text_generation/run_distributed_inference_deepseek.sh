#!/bin/bash
# Script to run distributed inference on Gaudi HPUs with Deepseek model

# Set the world size (number of HPUs to use)
WORLD_SIZE=8
export WORLD_SIZE

# Parameters for the model
MODEL_PATH="/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
INPUT_TEXT="Write a short story about artificial intelligence."
MAX_LENGTH=512

# Set up environment for Habana
export EXPERIMENTAL_WEIGHT_SHARING=true
export HABANA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPSPEED_HPU_ZERO_SHARDING_TYPE="fast_reduce"

# Remove any stale temporary files
rm -rf /tmp/deepspeed*

# Print info
echo "Starting distributed inference for Deepseek model"
echo "World size: $WORLD_SIZE"
echo "Model path: $MODEL_PATH"
echo "Input text: $INPUT_TEXT"

# Run using DeepSpeed launcher
cd $(dirname "$0")  # Move to the script directory

deepspeed --num_nodes=1 --num_gpus=$WORLD_SIZE \
    distributed_inference_deepseek.py \
    --model_path "$MODEL_PATH" \
    --input_text "$INPUT_TEXT" \
    --max_length $MAX_LENGTH

echo "Distributed inference completed"