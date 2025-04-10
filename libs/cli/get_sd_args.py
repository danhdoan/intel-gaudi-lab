"""CLI Parser.

Parse CLI arguments
"""

__author__ = "Nicolas Howard"
__email__ = "petit.nicolashoward@gmail.com"
__date__ = "2025/04/04"
__status__ = "development"


# ==============================================================================


import argparse

# ==============================================================================


def parse_args():
    """Parse CLI arguments from user.

    Returns
    -------
    (object) : parsed CLI arguments

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        type=str,
        help="Path to pre-trained model",
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
        "--timestep_spacing",
        default="linspace",
        choices=["linspace", "leading", "trailing"],
        type=str,
        help="The way the timesteps should be scaled.",
    )
    # Pipeline arguments
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="The file with prompts (for large number of images generation).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="An image of a squirrel in Picasso style",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts to guide the image generation \
            (applicable to SDXL).",
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
        "--height",
        type=int,
        default=0,
        help="The height in pixels of the generated images \
            (0=default from model config).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="The width in pixels of the generated images \
            (0=default from model config).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "The number of denoising steps. More denoising steps \
                usually lead to a higher quality image at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help=(
            "Guidance scale as defined in \
                [Classifier-Free Diffusion Guidance]\
                    (https://arxiv.org/abs/2207.12598)."
            " Higher guidance scale encourages to generate images \
                that are closely linked to the text `prompt`,"
            " usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default=None,
        help="The prompt or prompts not to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompts_2",
        type=str,
        nargs="*",
        default=None,
        help="The second prompt or prompts not to guide the image generation \
            (applicable to SDXL and SD3).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Corresponds to parameter eta (η) in the DDIM paper: \
            https://arxiv.org/abs/2010.02502.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--image_save_dir",
        type=str,
        default="./stable-diffusion-generated-images",
        help="The directory where images will be saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )

    # HPU-specific arguments
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
            "Name or path of the Gaudi configuration. \
                In particular, it enables to specify how to apply Habana Mixed"
            " Precision."
        ),
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="DeepSpeed local_rank argument for distributed running \
            will be set when running with DeepSpeed",
    )

    return parser.parse_args()


# ==============================================================================
