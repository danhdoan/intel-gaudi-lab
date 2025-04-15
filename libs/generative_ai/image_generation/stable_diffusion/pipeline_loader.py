"""Load the pipeline for Stable Diffusion models.

This module provides functions to load the appropriate
pipeline and scheduler for different Stable Diffusion models
based on the model name.
"""

from diffusers import DiffusionPipeline, SchedulerMixin

from .registry import PIPELINE_MAPPING, SCHEDULER_MAPPING

# ====================================================================


def select_scheduler(model_path: str) -> SchedulerMixin:
    """Load the scheduler based on the model name.

    Args:
    ----
        model_path (str): The name of the model to load the scheduler for.

    Returns:
    -------
        scheduler: The loaded scheduler.

    """
    scheduler_class = next(
        (
            scheduler
            for key, scheduler in SCHEDULER_MAPPING.items()
            if key in model_path
        ),
        SCHEDULER_MAPPING["default"],
    )
    return scheduler_class.from_pretrained(model_path, subfolder="scheduler")


# ====================================================================


def select_pipeline(model_path: str) -> DiffusionPipeline:
    """Load the pipeline based on the model name.

    Args:
    ----
        model_path (str): The name of the model to load the pipeline for.

    Returns:
    -------
        pipeline: The loaded pipeline.

    """
    pipeline_class = next(
        (
            model_type
            for key, model_type in PIPELINE_MAPPING.items()
            if key in model_path
        ),
        PIPELINE_MAPPING["default"],
    )
    return pipeline_class


# ====================================================================


def load_pipeline(
    model_path: str,
    use_habana: bool = True,
    use_hpu_graphs: bool = True,
    gaudi_config: str = "Habana/stable-diffusion",
) -> DiffusionPipeline:
    """Load the pipeline for the specified model.

    Args:
    ----
        model_path (str): The name of the model to load the pipeline for.
        use_habana (bool, optional): Whether to use Habana or not.
        use_hpu_graphs (bool, optional): Whether to use HPU graphs or not.
        gaudi_config (str, optional): The Gaudi configuration to use.

    Returns:
    -------
        pipeline: The loaded pipeline.

    """
    scheduler = select_scheduler(model_path)
    pipeline = select_pipeline(model_path)

    pipeline = pipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        use_habana=use_habana,
        use_hpu_graphs=use_hpu_graphs,
        gaudi_config=gaudi_config,
    )
    return pipeline
