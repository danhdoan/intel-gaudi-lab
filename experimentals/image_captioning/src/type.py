"""GenerateResponse models.

This module defines response models for image generation
using the Pydantic library.
"""

from pydantic import BaseModel

# ====================================================================


class GenerateResponse(BaseModel):
    """Response model for image generation.

    Args:
    ----
        answer (str): The generated answer.

    """

    answer: str
