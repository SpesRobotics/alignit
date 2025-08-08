# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

### Getting Started

```bash
# Record a dataset
python -m alignit.record

# Use the dataset to train a model
python -m alignit.train

# Visualize dataset
python -m alignit.visualize

# Run inference
python -m alignit.infere
```

### Development

```bash
# Install the package in editable mode
git clone https://github.com/SpesRobotics/alignit.git
cd alignit
pip install -e .
```

### Overriding Parameters

You can override defaults through the command line. For example, set the dataset path and model backbone during training:

```
python -m alignit.train --dataset.path=./data/duck --model.backbone=resnet18
```
