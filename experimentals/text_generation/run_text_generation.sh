#!/bin/bash

# Set default values
MODEL_PATH="/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PROMPT="What is artificial intelligence?"
MAX_NEW_TOKENS=500

# Function to run text generation
run_generation() {
    echo "Running text generation with parameters:"
    echo "Model: $1"
    echo "Prompt: $2"
    echo "Max new tokens: $3"
    echo "Use HPU graphs: $4"
    echo "----------------------------------------"

    # Build the command
    cmd="python text_generation_flow.py \
        --model_name_or_path \"$1\" \
        --prompt \"$2\" \
        --max_new_tokens \"$3\""

    # Add use_hpu_graphs flag if true
    if [ "$4" = true ]; then
        cmd="$cmd --use_hpu_graphs"
    fi

    # Execute the command
    eval "$cmd"
}

# Example 1: Basic generation
echo "Example 1: Basic text generation"
run_generation "$MODEL_PATH" "$PROMPT" "$MAX_NEW_TOKENS" true

# Example 2: Different prompt
echo -e "\nExample 2: Different prompt"
run_generation "$MODEL_PATH" "Explain quantum computing in simple terms." "$MAX_NEW_TOKENS" true

# Example 3: Shorter generation
echo -e "\nExample 3: Shorter generation"
run_generation "$MODEL_PATH" "Write a haiku about technology." 50 true

# Example 4: Without HPU graphs
echo -e "\nExample 4: Without HPU graphs"
run_generation "$MODEL_PATH" "Describe the benefits of renewable energy." "$MAX_NEW_TOKENS" false

echo -e "\nAll examples completed!"
