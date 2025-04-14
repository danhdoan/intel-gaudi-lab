"""Load the pipeline for Stable Diffusion models.

This module contains the function to load the pipeline for
Stable Diffusion models using the Optimum library.
"""

from optimum.habana.diffusers import GaudiI2VGenXLPipeline

# ==============================================================================


def load_pipeline(model_path: str) -> GaudiI2VGenXLPipeline:
    """Load Pipeline."""
    pipeline_kwargs = {
        "use_habana": True,
        "use_hpu_graphs": True,
        "gaudi_config": "Habana/stable-diffusion",
        "sdp_on_bf16": True,
        "bf16": True,
    }

    pipeline = GaudiI2VGenXLPipeline.from_pretrained(
        model_path,
        **pipeline_kwargs,
    )

    return pipeline
