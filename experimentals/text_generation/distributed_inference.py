import argparse
import os

import deepspeed
import torch
from habana_frameworks.torch.distributed import hccl
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

# Apply Gaudi adaptations
adapt_transformers_to_gaudi()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_rank",
    type=int,
    help="DeepSpeed local_rank argument for distributed running",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/intel-gaudi-lab/models/google-t5/t5-base",
    help="Path to the model",
)
parser.add_argument(
    "--input_text",
    type=str,
    default="Input String",
    help="Text for inference",
)
args = parser.parse_args()

# Get the world size from environment variable
world_size = int(os.getenv("WORLD_SIZE", "1"))
# world_size = 4
print(f"World size: {world_size}, Local rank: {args.local_rank}")

# Initialize HPU distributed environment
hccl.initialize_distributed_hpu(
    world_size=world_size, local_rank=args.local_rank
)
torch.distributed.init_process_group(
    backend="hccl", world_size=world_size, rank=args.local_rank
)

# Prepare DeepSpeed arguments
deepspeed_parser = argparse.ArgumentParser()
deepspeed_args = deepspeed_parser.parse_args(args="")

# Load model configuration, tokenizer, and model using the transformers library
print(f"Loading model from {args.model_path}")
config = AutoConfig.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    use_fast=True,
    trust_remote_code=True,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    args.model_path,
    from_tf=bool(".ckpt" in args.model_path),
    config=config,
    trust_remote_code=True,
)

# Create the pipeline with proper model
pipe = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer, device="hpu"
)

# Initialize the DeepSpeed inference engine
pipe.model = deepspeed.init_inference(
    pipe.model,
    tensor_parallel={"tp_size": world_size},
    dtype=torch.float,
    kernel_inject=True,
    # injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}, # No longer needed
)

# Generate output
print(f"Generating text for input: '{args.input_text}'")
output = pipe(args.input_text)
print(f"Generated output: {output[0]['generated_text']}")
# print(f"Generated output: {output[0]}")
