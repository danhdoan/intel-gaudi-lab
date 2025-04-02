"""Stable Diffusion module."""

import argparse
import time
from pathlib import Path

import torch
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiStableDiffusionPipeline,
)
from optimum.habana.utils import set_seed

# ==============================================================================


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Inference Parameters"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="peaceful beach, horizon, wide view, master works, HD, HQ, 4k",
        help="The prompt to generate images for.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurred, paintings, sketches, lowres, unnatural",
        help="The negative prompt to guide image generation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of images in a batch.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to run.",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the output image."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the output image."
    )
    parser.add_argument(
        "--scheduler",
        default="ddim",
        choices=[
            "default",
            "euler_discrete",
            "euler_ancestral_discrete",
            "ddim",
            "flow_match_euler_discrete",
        ],
        type=str,
        help="Name of scheduler",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="models/stable-diffusion-xl-base-1.0",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Use HPU graphs on HPU. This should lead to faster generations.",
    )
    parser.add_argument(
        "--gaudi_config_name",
        type=str,
        default="Habana/stable-diffusion",
        help="Name or path of the Gaudi configuration.",
    )

    return parser.parse_args()


# ==============================================================================


def app(args):
    """Perform main logic."""
    set_seed(args.seed)

    # Configure scheduler based on args
    kwargs = {"timestep_spacing": "linspace"}
    scheduler = GaudiDDIMScheduler.from_pretrained(
        args.model_name_or_path, subfolder="scheduler", **kwargs
    )

    # Configure pipeline options
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    pipeline_kwargs = {
        "scheduler": scheduler,
        "torch_dtype": kwargs["torch_dtype"],
    }

    # Initialize pipeline
    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        **pipeline_kwargs,
        gaudi_config=args.gaudi_config_name,
        use_habana=args.use_habana,
        use_hpu_graphs=args.use_hpu_graphs,
    )

    # Configure generation parameters
    generation_kwargs = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
    }

    output = pipeline(**generation_kwargs)

    # Save images
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    for i, image in enumerate(output.images):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{timestamp}-image_{i}.png"
        image.save(filename)


# ==============================================================================


def main():
    """Perform main logic."""
    args = parse_args()
    app(args)


# ==============================================================================

if __name__ == "__main__":
    main()
