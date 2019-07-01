from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="jalef",
    version="0.0.1",
    author="Márton Torner",
    author_email="torner.marton@gmail.com",
    description="Just Another Language Engineering Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tornermarton/jalef",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tensorflow'
    ],
)