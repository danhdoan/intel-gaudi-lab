#!/bin/bash
# Script to run distributed inference on Gaudi HPUs

# Set the world size (number of HPUs to use)
WORLD_SIZE=8
export WORLD_SIZE

# Parameters for the model
MODEL_PATH="/intel-gaudi-lab/models/google-t5/t5-base"
INPUT_TEXT="Translate this to German: Can I invite you to my party?"

# Set up environment for Habana
export EXPERIMENTAL_WEIGHT_SHARING=true
export HABANA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPSPEED_HPU_ZERO_SHARDING_TYPE="fast_reduce"

# Remove any stale temporary files
rm -rf /tmp/deepspeed*

# Print info
echo "Starting distributed inference with world size: $WORLD_SIZE"
echo "Model path: $MODEL_PATH"
echo "Input text: $INPUT_TEXT"

# Run using DeepSpeed launcher
deepspeed --num_gpus 1 distributed_inference.py \
    --model_path "$MODEL_PATH" \
    --input_text "$INPUT_TEXT"
