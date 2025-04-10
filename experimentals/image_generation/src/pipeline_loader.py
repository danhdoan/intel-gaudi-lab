"""Load the pipeline for Stable Diffusion models.

This module provides functions to load the appropriate
pipeline and scheduler for different Stable Diffusion models
based on the model name.
"""

from diffusers import DiffusionPipeline, SchedulerMixin
from src.macro import PIPELINE_MAPPING, SCHEDULER_MAPPING

# ====================================================================


def select_scheduler(model_name: str) -> SchedulerMixin:
    """Load the scheduler based on the model name.

    Args:
    ----
        model_name (str): The name of the model to load the scheduler for.

    Returns:
    -------
        scheduler: The loaded scheduler.

    """
    scheduler_class = next(
        (
            scheduler
            for key, scheduler in SCHEDULER_MAPPING.items()
            if key in model_name
        ),
        SCHEDULER_MAPPING["default"],
    )
    return scheduler_class.from_pretrained(model_name, subfolder="scheduler")


# ====================================================================


def select_pipeline(model_name: str) -> DiffusionPipeline:
    """Load the pipeline based on the model name.

    Args:
    ----
        model_name (str): The name of the model to load the pipeline for.

    Returns:
    -------
        pipeline: The loaded pipeline.

    """
    pipeline_class = next(
        (
            model_type
            for key, model_type in PIPELINE_MAPPING.items()
            if key in model_name
        ),
        PIPELINE_MAPPING["default"],
    )
    return pipeline_class


# ====================================================================


def load_pipeline(
    model_name: str,
) -> DiffusionPipeline:
    """Load the pipeline for the specified model.

    Args:
    ----
        model_name (str): The name of the model to load the pipeline for.

    Returns:
    -------
        pipeline: The loaded pipeline.

    """
    scheduler = select_scheduler(model_name)
    pipeline = select_pipeline(model_name)

    pipeline = pipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion",
    )
    return pipeline
