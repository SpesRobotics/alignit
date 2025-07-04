from setuptools import setup, find_packages

setup(
    name="alignit",
    version="0.0.1",
    description="Real-time robot arm alignment using computer vision",
    author="Darko Lukic",
    author_email="darko.lukic@spes.ai",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
)
