#!/usr/bin/env python
import setuptools
import os

os.chmod("run.py", 0o744)
os.chmod("bsm.py", 0o744)

setuptools.setup(
    name='dHDNNcontrollers',
    version='1.0',
    url='https://github.com/DecodEPFL/dHDNNcontrollers',
    license='',
    author='Clara Galimberti',
    author_email='clara.galimberti@epfl.ch',
    description='Distributed H-DNN controllers',
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.7.1',
                      'numpy>=1.18.1',
                      'matplotlib>=3.1.3',
                      'torchvision>=0.5.0',
                      'torchdiffeq>=0.2.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
