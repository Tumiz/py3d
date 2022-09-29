#!/usr/bin/env python3
from setuptools import setup, find_packages
import urllib.request
import json

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


setup(
    name="py3d",
    version=next_version(latest_version),
    description="py3d is a python 3d computational geometry library, which can deal with points, lines, planes and 3d meshes in batches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tumiz",
    author_email="hh11698@163.com",
    python_requires=">=3.7.0",
    url="https://tumiz.github.io/scenario/",
    install_requires=["numpy", "ipython"],
    packages=find_packages(),
    data_files=[
        ("/py3d", ["py3d/viewer.html", "py3d/car.npy"])
    ],
    license="GPL-3.0 License"
)
