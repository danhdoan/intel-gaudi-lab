"""Process Image Module.

This module contains functions for processing images, including converting
images to base64 format.
"""

import base64
from io import BytesIO

# ======================================================================


def image_to_base64(images: list[str]) -> list[str]:
    """Convert images to base64 format.

    Args:
        images (list[str]): List of images to be converted.

    Returns:
        list[str]: List of base64 encoded images.

    """
    list_base64 = []
    for i in range(len(images)):
        buffered = BytesIO()
        images[i].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images[i] = img_str
        list_base64.append(img_str)
    return list_base64


# ======================================================================
