from setuptools import setup, find_packages

setup(
    name="alignit",
    version="0.0.1",
    description="Model-free real-time robot arm alignment using one or more RGB(D) cameras.",
    author="Spes Robotics",
    author_email="contact@spes.ai",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "datasets",
        "gradio",
        "transforms3d",
        "pybullet",
        "tqdm",
        "matplotlib",
        "mujoco",
        "numpy",
        "teleop[utils]",
        "xarm-python-sdk",
        "lerobot @ git+https://github.com/huggingface/lerobot.git@67196c9"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
)
