#!/usr/bin/env python3

import toweb
from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="toweb",
    version="0.0.2",
    description="a library help you show logs and data on web",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.5.0",
    url="https://github.com/Tumiz/scenario",
    install_requires=["tornado"],
    packages=find_packages(),
    data_files=[
        ("/toweb/static",["toweb/static/logo.png","toweb/static/bundle.js"]),
        ("/toweb",["toweb/viewer.html"])
        ],
    license="GPL-3.0 License"
)
