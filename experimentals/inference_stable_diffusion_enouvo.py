import argparse
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType

from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiStableDiffusionPipeline
)
from optimum.habana.utils import set_seed
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker


def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion Inference Parameters')
    
    # Add the specific arguments we want to parse
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of an astronaut riding a horse on mars",
        help="The prompt to generate images for."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization."
    )
    
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of images in a batch."
    )
    
    parser.add_argument(
        "--scheduler",
        default="ddim",
        choices=["default", "euler_discrete", "euler_ancestral_discrete", "ddim", "flow_match_euler_discrete"],
        type=str,
        help="Name of scheduler"
    )
    
    parser.add_argument(
        "--model_name_or_path",
        default="models/stable-diffusion-2-base",
        type=str,
        help="Path to pre-trained model"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    # safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

    # Configure scheduler based on args
    kwargs = {"timestep_spacing": "linspace"}
    if args.scheduler == "flow_match_euler_discrete":
        scheduler = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "euler_discrete":
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        ) 
    elif args.scheduler == "ddim":
        scheduler = GaudiDDIMScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    else:
        scheduler = None

    # Configure pipeline options
    pipeline_kwargs = {
        "scheduler": scheduler,
        "torch_dtype": torch.float32
    }

    # Initialize pipeline
    pipeline = GaudiStableDiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        **pipeline_kwargs,
        gaudi_config="Habana/stable-diffusion",
        use_habana=True,
        use_hpu_graphs=True,
        # safety_checker=safety_checker,
    )

    # Configure generation parameters
    generation_kwargs = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "prompt": args.prompt,
    }

    output = pipeline(**generation_kwargs)

    # Save images
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    for i, image in enumerate(output.images):
        image.save(output_dir / f"image_{i}.png")
if __name__ == "__main__":
    main()