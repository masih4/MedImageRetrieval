from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='uni',
    version='0.1.0',
    description='UNI',
    url='https://github.com/mahmoodlab/UNI',
    author='RJC,MYL,TD',
    author_email='',
    license='CC BY-NC 4.0',
    packages=find_packages(exclude=['__dep__', 'assets']),
    install_requires=["timm==0.9.8", "numpy", "pandas", "scikit-learn", "tqdm", "transformers", "opencv-python"],

    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: CC BY-NC 4.0",
]
)
