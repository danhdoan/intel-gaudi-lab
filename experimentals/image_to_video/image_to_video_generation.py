"""Experiment Image to Video."""

import argparse
import os

import torch
from diffusers.utils import export_to_video, load_image
from optimum.habana.diffusers import GaudiI2VGenXLPipeline
from optimum.habana.utils import set_seed

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
        "--prompts",
        type=str,
        nargs="*",
        default="",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default="",
        help="The prompt or prompts not to guide the image generation.",
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
        "--num_frames",
        type=int,
        default=25,
        help="The number of video frames to generate.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="The number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="The guidance scale.",
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
        "--fps",
        type=int,
        default=7,
        help="Frames per second.",
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
    pipeline_kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }
    if args.bf16:
        pipeline_kwargs["torch_dtype"] = torch.bfloat16

    pipeline = GaudiI2VGenXLPipeline.from_pretrained(
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
    generator = torch.manual_seed(args.seed)
    generation_kwargs = {
        "prompt": args.prompts,
        "num_videos_per_prompt": args.num_videos_per_prompt,
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "negative_prompt": args.negative_prompts,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }

    return generation_kwargs


# ==============================================================================


def save_outputs(outputs, args):
    """Save outputs."""
    for i, frames in enumerate(outputs.frames):
        export_to_video(
            frames,
            f"{args.output_path}/gen_video_{str(i).zfill(2)}.mp4",
            fps=args.fps,
        )


# ==============================================================================


def app(args):
    """Perform main logic."""
    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.seed)

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
