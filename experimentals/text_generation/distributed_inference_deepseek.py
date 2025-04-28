import argparse
import logging
import os
import tempfile

import deepspeed
import torch
from habana_frameworks.torch.distributed import hccl
from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    write_checkpoints_json,
)
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Apply Gaudi adaptations
adapt_transformers_to_gaudi()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_rank",
    type=int,
    default=0,
    help="DeepSpeed local_rank argument for distributed running",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/intel-gaudi-lab/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    help="Path to the model",
)
parser.add_argument(
    "--input_text",
    type=str,
    default="What is deep learning?",
    help="Text for inference",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Maximum length of generated text",
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="Enable streaming output",
)
args = parser.parse_args()

# Get distributed environment information
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = args.local_rank
logger.info(f"World size: {world_size}, Local rank: {local_rank}")

# Initialize HPU distributed environment
hccl.initialize_distributed_hpu(
    world_size=world_size, rank=local_rank, local_rank=local_rank
)
torch.distributed.init_process_group(backend="hccl")

# Load model configuration
logger.info(f"Loading model configuration from {args.model_path}")
model_config = AutoConfig.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
    use_cache=True,
)

# Load and configure the tokenizer
logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")
    else:
        tokenizer.pad_token = tokenizer.eos_token = "</s>"
        logger.info(
            "No eos_token found, setting pad_token and eos_token to '</s>'"
        )

# Create a model with meta tensors
logger.info("Creating model with meta tensors")
with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
    model = AutoModelForCausalLM.from_config(
        model_config, torch_dtype=torch.bfloat16
    )
model = model.eval()

# Create a file to indicate where the checkpoint is
logger.info("Writing checkpoint information")
checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="w+")
write_checkpoints_json(args.model_path, local_rank, checkpoints_json, token="")

# Prepare the DeepSpeed inference configuration
logger.info("Preparing DeepSpeed inference configuration")
kwargs = {"dtype": torch.bfloat16}
kwargs["checkpoint"] = checkpoints_json.name
kwargs["tensor_parallel"] = {"tp_size": world_size}
# Enable the HPU graph, similar to the cuda graph
kwargs["enable_cuda_graph"] = True
# Specify the injection policy, required by DeepSpeed Tensor parallelism
kwargs["injection_policy"] = get_ds_injection_policy(model_config)

# Initialize the DeepSpeed inference engine
logger.info("Initializing DeepSpeed inference engine")
model = deepspeed.init_inference(model, **kwargs).module


# Define a custom streamer for text output if streaming is enabled
class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_special_tokens=True):
        super().__init__(tokenizer, skip_special_tokens=skip_special_tokens)
        self.text_so_far = ""

    def on_finalized_text(self, text, stream_end=False):
        print(text, end="", flush=True)
        self.text_so_far += text

    def get_text(self):
        return self.text_so_far


# Function to tokenize the input and move it to HPU
def tokenize(prompt):
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True)
    return input_tokens.input_ids.to(device="hpu")


# Generate text
if local_rank == 0:
    logger.info(f"Generating text for input: '{args.input_text}'")

# Create generation parameters
generation_config = {
    "max_length": args.max_length,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# Process based on whether streaming is enabled
if args.streaming and local_rank == 0:
    # For streaming, we use the TextStreamer
    streamer = CustomTextStreamer(tokenizer, skip_special_tokens=True)

    # Tokenize input
    input_ids = tokenize(args.input_text)

    # Generate with streamer
    print("\nGenerated output (streaming):")
    model.generate(input_ids, streamer=streamer, **generation_config)
    print("\n")  # Add a newline after generation
    generated_text = streamer.get_text()
else:
    # For non-streaming or non-rank-0 processes
    input_ids = tokenize(args.input_text)

    # Generate without streamer
    with torch.no_grad():
        outputs = model.generate(input_ids, **generation_config)

    # Decode outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print results only on rank 0 to avoid duplicate output
    if local_rank == 0:
        print("\nGenerated output:")
        print(generated_text)
        print("\n")

# Clean up
checkpoints_json.close()

# Log completion
if local_rank == 0:
    logger.info("Generation completed successfully")
