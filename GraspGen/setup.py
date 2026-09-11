# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pathlib import Path
from setuptools import setup, find_packages


def _read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return []
    return [line for line in req_path.read_text().splitlines() if "@" not in line and line.strip()]

setup(
    name="grasp_gen",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=_read_requirements(),
    description="GraspGen",
    author="",
    author_email="",
    license="",
    url="",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
