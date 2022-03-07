#!/usr/bin/env python

# PyTropD installation script
from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(_this_directory_, 'README')) as f:
      long_description = f.read()

setup (name="pytropd",
	version='1.0.5',
        description = "Calculation of metrics of tropical width",
	long_description=_long_description_,
        license = "GPL-3",
        author="Alison Ming, Paul William Staten, Samuel Smith",
        author_email="admg26@gmail.com",
        url="https://tropd.github.io/pytropd/index.html",
	install_requires=['numpy>=1.19','scipy>=1.5','python_version>=3.6'],
	packages=["pytropd"],
        classifiers=["Programming Language :: Python :: 3"],
)

