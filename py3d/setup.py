#!/usr/bin/env python3

from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="py3d",
    version="0.0.15",
    description="a 3d library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.5.0",
    url="https://github.com/Tumiz/scenario",
    install_requires=["numpy","tornado","ipython"],
    packages=find_packages(),
    data_files=[
        ("/py3d/static",["py3d/static/logo.png","py3d/static/bundle.js"]),
        ("/py3d",["py3d/viewer.html"])
        ],
    license="GPL-3.0 License"
)
