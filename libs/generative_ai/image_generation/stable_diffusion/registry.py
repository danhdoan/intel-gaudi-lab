"""Module containing the Gaudi Diffusers pipeline and schedulers."""

from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
    GaudiStableDiffusion3Pipeline,
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionXLPipeline,
)

SCHEDULER_MAPPING = {
    "stable-diffusion-3": GaudiFlowMatchEulerDiscreteScheduler,
    "default": GaudiDDIMScheduler,
}

PIPELINE_MAPPING = {
    "stable-diffusion-xl": GaudiStableDiffusionXLPipeline,
    "stable-diffusion-3": GaudiStableDiffusion3Pipeline,
    "default": GaudiStableDiffusionPipeline,
}

DIFFUSION_MODEL = [
    "stable-diffusion-2-base",
    "stable-diffusion-2.1",
    "stable-diffusion-3-m-d",
    "stable-diffusion-xl-base-1.0",
]

MODEL_PATH_FOLDER = "/path/to/model_folder"  # e.g., "/intel-gaudi-lab/models"
