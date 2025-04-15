"""GenerateRequest and GenerateResponse models.

This module defines the request and response models for image generation
using the Pydantic library.
"""

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request model for image generation.

    Args:
    ----
        prompt (str): The text prompt to generate the image.
        negative_prompt (str | None): The negative prompt
        num_inference_steps (int): The number of inference steps to take.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        num_images_per_prompt (int): The number of images to generate
        batch_size (int): The number of images to generate in a batch.
        guidance_scale (float): The scale for classifier-free guidance.
        seed (int): The random seed for reproducibility.

    """

    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int = 30
    nums_frames: int = 30
    num_video_per_prompt: int = 1
    batch_size: int = 1
    guidance_scale: float = 7.5
    fps: int = 7
    seed: int = 42


# ====================================================================


class GenerateResponse(BaseModel):
    """Response model for image generation.

    Args:
    ----
        image (str): The generated list of base64 encoded images.

    """

    image: list[str]
