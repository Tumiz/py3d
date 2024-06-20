#!/usr/bin/env python3
from setuptools import setup, find_packages
import urllib.request
import json
import sys

rep = urllib.request.urlopen("https://pypi.org/pypi/py3d/json")
rep_dict = json.loads(rep.read().decode())
latest_version = rep_dict["info"]["version"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def next_version(s):
    vs = s.split('.')
    i = 2
    while True:
        v = int(vs[i])
        if v + 1 <= 99:
            vs[i] = str(v + 1)
            break
        else:
            vs[i] = '0'
            i -= 1
    return '.'.join(vs)

def ipython_version(python_version):
    if python_version==7:
        return "7.*"
    elif python_version==8:
        return ">=8.0,<=8.12"
    else:
        return ">=8.13"

setup(
    name="py3d",
    version=next_version(latest_version),
    description="py3d is a pure and lightweight python library of 3d data structures and functions, which can deal with points, lines, planes and 3d meshes in batches, and also visualize them. All the actions can be done in a jupyter notebook.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.7.0",
    url="https://tumiz.github.io/py3d/",
    install_requires=["pillow", "numpy", "ipython{}".format(ipython_version(sys.version_info.minor))],
    packages=find_packages(),
    data_files=[
        ("/py3d", ["py3d/viewer.html"])
    ],
    license="GPL-3.0 License"
)
