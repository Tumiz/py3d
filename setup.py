#!/usr/bin/env python3

from setuptools import setup

setup(
    name="scenario",
    description="a 3d library, spatial-temporal analysis tool",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.5.0",
    url="https://github.com/Tumiz/scenario",
    install_requires=["tornado","numpy"],
    packages=["scenario"]
)