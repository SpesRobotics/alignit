import re
from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Replace relative links with absolute links
for match in re.findall(r"\]\((?!http)([^)]+)\)", long_description):
    filepath = Path(match)
    long_description = long_description.replace(
        match, f"https://github.com/SpesRobotics/alignit/raw/main/{filepath}"
    )


setup(
    name="alignit",
    version="0.0.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Model-free real-time robot arm alignment using one or more RGB(D) cameras.",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "datasets",
        "gradio",
        "transforms3d",
        "tqdm",
        "matplotlib",
        "mujoco",
        "numpy",
        "teleop[utils]",
        "xarm-python-sdk",
        "draccus",
        "lerobot",
    ],
    license="Apache 2.0",
    python_requires=">=3.8",
    author="Spes Robotics",
    author_email="contact@spes.ai",
    project_urls={
        "Documentation": "https://github.com/SpesRobotics/alignit",
        "Source": "https://github.com/SpesRobotics/alignit",
        "Tracker": "https://github.com/SpesRobotics/alignit/issues",
    },
)
