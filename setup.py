"""Setup `libs` as stand-alone library."""

__author__ = ["Danh Doan", "Nam Ho"]
__email__ = ["danh.doan@enouvo.com", "nam.ho@enouvo.com"]
__date__ = "2025/04/01"
__status__ = "development"


# ==============================================================================


from setuptools import find_packages, setup


# ==============================================================================


setup(
    name="libs",
    version="1.0.0",
    description="Internal Module for Development",
    author="Enouvo-AI",
    author_email="",
    packages=find_packages(),
    setup_requires=["setuptools", "wheel"],
)


# ==============================================================================
