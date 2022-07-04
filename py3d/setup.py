#!/usr/bin/env python3

from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py3d",
    version="0.0.28",
    description="py3d is a python 3d computational geometry library, which can deal with points, lines, planes and 3d meshes in batches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.7.0",
    url="https://tumiz.github.io/scenario/",
    install_requires=["numpy","tornado","ipython"],
    packages=find_packages(),
    data_files=[
        ("/py3d/static",["py3d/static/bundle.js"]),
        ("/py3d",["py3d/viewer.html"])
        ],
    license="GPL-3.0 License"
)
