"""Text generation command-line utility for Intel Gaudi accelerators.

This module provides a simple command-line interface for text generation using
large language models on Intel Gaudi HPUs. It handles model loading,
inference, and output formatting.
"""

import argparse
import logging

import torch
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from optimum.habana.utils import get_hpu_memory_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
adapt_transformers_to_gaudi()


class TextGenerator:
    """Manages text generation using a language model on Gaudi accelerators."""

    def __init__(
        self,
        model_name_or_path,
        device="hpu",
        max_new_tokens=500,
        use_hpu_graphs=True,
    ):
        """Init the text generator with the specified model and parameters.

        Args:
            model_name_or_path: Path or name of the pre-trained model
            device: Device to run inference on (default: "hpu")
            max_new_tokens: Maximum number of new tokens to generate
            use_hpu_graphs: Whether to use HPU graphs for optimization

        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_hpu_graphs = use_hpu_graphs

        # Initialize model and tokenizer
        logger.info(f"Loading model from {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Set padding side
        if not self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # Move model to device
        self.model = self.model.eval().to(self.device)

        # Wrap model with HPU graphs if enabled
        if self.use_hpu_graphs:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            self.model = wrap_in_hpu_graph(self.model)

        logger.info("Model initialization completed")

    def generate(self, prompt):
        """Generate text from a prompt.

        Args:
            prompt: The input text to base generation on

        Returns:
            The generated text as a string

        """
        # Tokenize input
        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)

        # Move inputs to device
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        # Generate text
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            lazy_mode=True,
            hpu_graphs=self.use_hpu_graphs,
        )

        # Decode and return generated text
        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return generated_text[0]


def main():
    """Run the text generation command-line utility.

    Parses command-line arguments, initializes the generator,
    performs text generation, and displays results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="hpu",
        choices=["hpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs",
    )

    args = parser.parse_args()

    # Initialize text generator
    generator = TextGenerator(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        use_hpu_graphs=args.use_hpu_graphs,
    )

    # Generate text
    generated_text = generator.generate(args.prompt)

    # Print results
    print("\nInput prompt:", args.prompt)
    print("\nGenerated text:", generated_text)

    # Print memory stats
    mem = get_hpu_memory_stats()
    print("\nMemory stats:")
    for k, v in mem.items():
        print(f"{k[:-5].replace('_', ' ').capitalize():35} = {v} GB")


if __name__ == "__main__":
    main()
