"""Stable Diffusion Inference Pipeline for Gaudi."""

import os

from optimum.habana import utils as habana_utils

from .pipeline_loader import load_pipeline
from .registry import MODEL_PATH_FOLDER

# ====================================================================


class GaudiStableDiffusionInferencePipeline:
    """A class for running Stable Diffusion inference."""

    def __init__(
        self,
        model_name: str = "stable-diffusion-2-base",
        use_habana: bool = True,
        use_hpu_graphs: bool = True,
        gaudi_config: str = "Habana/stable-diffusion",
    ):
        """Initialize the pipeline with a pre-trained Stable Diffusion model.

        Args:
        ----
            model_name (str): The path to the pre-trained model.
            use_habana (bool, optional): Whether to use Habana or not.
            use_hpu_graphs (bool, optional): Whether to use HPU graphs or not.
            gaudi_config (str, optional): The Gaudi configuration to use.

        Returns:
        -------
            None

        """
        model_path = os.path.join(
            MODEL_PATH_FOLDER,
            model_name,
        )
        self.pipe = load_pipeline(
            model_path,
            use_habana=use_habana,
            use_hpu_graphs=use_hpu_graphs,
            gaudi_config=gaudi_config,
        )

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = 42,
        num_images_per_prompt: int = 1,
        batch_size: int = 1,
    ) -> list:
        """Generate an image based on the given prompt.

        Args:
        ----
            prompt (str): The prompt to generate the image from.
            negative_prompt (str, optional): The negative prompt to avoid.
            num_inference_steps (int, optional): The number of inference steps.
            guidance_scale (float, optional): The guidance scale.
            height (int, optional): The height of the generated image.
            width (int, optional): The width of the generated image.
            seed (int, optional): The seed for random number generation.
            num_images_per_prompt (int, optional): Number of images per prompt.
            batch_size (int, optional): Batch size for generation.

        Returns:
        -------
            list: A list of generated images.

        """
        habana_utils.set_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
        )

        return output.images

    def change_model(
        self,
        model_name: str = "stable-diffusion-2-base",
        use_habana: bool = True,
        use_hpu_graphs: bool = True,
        gaudi_config: str = "Habana/stable-diffusion",
    ):
        """Remove previous pipeline and reinitialize the pipeline.

        Args:
        ----
            model_name (str): The path to the pre-trained model.
            use_habana (bool, optional): Whether to use Habana or not.
            use_hpu_graphs (bool, optional): Whether to use HPU graphs or not.
            gaudi_config (str, optional): The Gaudi configuration to use.

        Returns:
        -------
            None

        """
        del self.pipe
        model_path = os.path.join(
            MODEL_PATH_FOLDER,
            model_name,
        )
        self.pipe = load_pipeline(
            model_path,
            use_habana=use_habana,
            use_hpu_graphs=use_hpu_graphs,
            gaudi_config=gaudi_config,
        )


# ===================================================================
