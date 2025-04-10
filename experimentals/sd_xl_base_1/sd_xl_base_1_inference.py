"""Inference Application.

with Stable Diffusion XL Base 1.0 Model
"""

__author__ = ["Nicolas Howard", "Nguyen Tran"]
__email__ = ["petit.nicolashoward@gmail.com", "nguyen.tran@enouvo.com"]
__date__ = "2025/04/04"
__status__ = "development"


# ==============================================================================


from pathlib import Path

import torch
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiStableDiffusionXLPipeline,
)
from optimum.habana.utils import set_seed

from libs.cli.get_sd_args import parse_args
from libs.utils import logger_utils
from libs.utils.time_utils import tiktok

# ==============================================================================


# Setup logger
logger = logger_utils.setup_logger(name=__name__, log_dir="logs")


# ==============================================================================


def get_scheduler(args, scheduler_kwargs):
    """Get scheduler."""
    if args.scheduler == "euler_discrete":
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **scheduler_kwargs
        )
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **scheduler_kwargs
        )
    elif args.scheduler == "ddim":
        scheduler = GaudiDDIMScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **scheduler_kwargs
        )
    else:
        scheduler = None

    return scheduler


# ==============================================================================


def get_prompts(args, kwargs_call):
    """Get prompts."""
    # If prompts file is specified override prompts from the file
    if args.prompts_file is not None:
        lines = []
        with open(args.prompts_file) as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        args.prompts = lines

    kwargs_call["negative_prompt"] = args.negative_prompts
    kwargs_call["prompt_2"] = args.prompts_2
    kwargs_call["negative_prompt_2"] = args.negative_prompts_2


# ==============================================================================


def get_pipeline(args, kwargs):
    """Get pipeline."""
    pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    return pipeline


# ==============================================================================


def generate_outputs(args, pipeline, kwargs_call):
    """Generate outputs."""
    # Generate Images using a Stable Diffusion pipeline
    outputs = pipeline(prompt=args.prompts, **kwargs_call)

    return outputs


# ==============================================================================


@tiktok
def inference(args, pipeline, kwargs_call):
    """Inference function."""
    for _ in range(10):
        _outputs = pipeline(prompt=args.prompts, **kwargs_call)


# ==============================================================================


def save_images(args, outputs):
    """Save pipeline's outputs.

    Save images in the specified directory if not None and if they are in
    PIL format.
    """
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)

            image_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {image_save_dir.resolve()}...")

            for i, image in enumerate(outputs.images):
                image.save(image_save_dir / f"image_{i + 1}.png")
        else:
            logger.warning(
                """
                --output_type should be equal to 'pil' to save images
                in --image_save_dir.
                """
            )


# ==============================================================================


def app(args):
    """Perform main logics of the application."""
    # Set RNG seed
    set_seed(args.seed)

    # Set the scheduler
    scheduler_kwargs = {"timestep_spacing": args.timestep_spacing}
    scheduler = get_scheduler(args, scheduler_kwargs)

    # Set pipeline class instantiation options
    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }

    if scheduler is not None:
        kwargs["scheduler"] = scheduler
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    # Set pipeline call options
    kwargs_call = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "output_type": args.output_type,
    }

    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height

    pipeline = get_pipeline(args, kwargs)
    get_prompts(args, kwargs_call)
    outputs = generate_outputs(args, pipeline, kwargs_call)
    save_images(args, outputs)

    # * Performance benchmark: Inference Time x10 *
    get_prompts(args, kwargs_call)
    inference(args, pipeline, kwargs_call)


# ==============================================================================


def main():
    """Perform CLI parse and run the logics."""
    args = parse_args()
    app(args)


# ==============================================================================


if __name__ == "__main__":
    main()


# ==============================================================================
