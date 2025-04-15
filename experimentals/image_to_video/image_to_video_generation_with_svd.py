"""Experiment Image to Video."""

import argparse
import os

import torch
from diffusers.utils import export_to_video, load_image, export_to_gif
from optimum.habana.diffusers import (
    GaudiStableVideoDiffusionPipeline,
    GaudiEulerDiscreteScheduler
)
from optimum.habana import utils as habana_utils

# ==============================================================================


def get_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="ali-vilab/i2vgen-xl",
        help="Path to pre-trained model",
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
        help=(
            "Name or path of the Gaudi configuration. In particular, "
            "it enables to specify how to apply Habana Mixed Precision."
        ),
    )
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        default=False,
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )

    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="The number of videos to generate per prompt image.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of videos in a batch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="The height in pixels of the generated video."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width in pixels of the generated video."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.0,
        help=(
            "The minimum guidance scale. "
            "Used for the classifier free guidance with last frame."
        ),
    )
    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
        help=(
            "The maximum guidance scale. "
            "Used for the classifier free guidance with last frame."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="Frames per second.",
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
        help=(
            "The motion bucket ID. Used as conditioning for the generation. "
            "The higher the number the more motion will be in the video."
        ),
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help=(
            "The amount of noise added to the init image, the higher it is "
            "the less the video will look like the init image. "
            "Increase it for more motion."
        ),
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=None,
        help=(
            "The number of frames to decode at a time. "
            "The higher the chunk size, the higher the temporal consistency "
            "between frames, but also the higher the memory consumption. "
            "By default, the decoder will decode all frames at once for "
            "maximal quality. "
            "Reduce `decode_chunk_size` to reduce memory usage."
        ),
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to ignore for throughput calculation.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image(s) to guide video generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
        help="Path to output folder",
    )

    return parser.parse_args()


# ==============================================================================


def create_pipeline(args):
    """Create Pipeline."""
    scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )
    pipeline_kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }
    if args.bf16:
        pipeline_kwargs["torch_dtype"] = torch.bfloat16

    pipeline = GaudiStableVideoDiffusionPipeline.from_pretrained(
        args.model_path,
        **pipeline_kwargs,
    )

    return pipeline


# ==============================================================================


def load_input_images(image_path):
    """Load input images from given image folder."""
    input_images = []
    fnames = os.listdir(image_path)
    for fname in fnames:
        fpath = os.path.join(image_path, fname)
        image = load_image(fpath)
        image = image.convert("RGB")
        input_images.append(image)

    return input_images


# ==============================================================================


def define_generation_kwargs(args):
    """Define keyword-argument pairs for generation."""
    generation_kwargs = {
        "num_videos_per_prompt": args.num_videos_per_prompt,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "min_guidance_scale": args.min_guidance_scale,
        "max_guidance_scale": args.max_guidance_scale,
        "fps": args.fps,
        "motion_bucket_id": args.motion_bucket_id,
        "noise_aug_strength": args.noise_aug_strength,
        "decode_chunk_size": args.decode_chunk_size,
        "output_type": args.output_type,
        "profiling_warmup_steps": args.profiling_warmup_steps,
        "profiling_steps": args.profiling_steps,
    }
    if args.throughput_warmup_steps is not None:
        generation_kwargs[
            "throughput_warmup_steps"
        ] = args.throughput_warmup_steps

    return generation_kwargs


# ==============================================================================


def save_outputs(outputs, args):
    """Save outputs."""
    for i, frames in enumerate(outputs.frames):
        export_to_gif(
            frames,
            f"{args.output_path}/gen_video_{str(i).zfill(2)}.gif",
        )


# ==============================================================================


def app(args):
    """Perform main logic."""
    os.makedirs(args.output_path, exist_ok=True)
    habana_utils.set_seed(args.seed)

    pipeline = create_pipeline(args)
    input_images = load_input_images(args.image_path)
    generation_kwargs = define_generation_kwargs(args)

    outputs = pipeline(image=input_images, **generation_kwargs)

    save_outputs(outputs, args)


# ==============================================================================


def main():
    """Perform main logic."""
    args = get_args()
    app(args)


# ==============================================================================


if __name__ == "__main__":
    main()


# ==============================================================================
