"""Logging configuration for text generation modules.

This module sets up the logging configuration used by the text generation
components, ensuring consistent log formatting across the application.
"""

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
