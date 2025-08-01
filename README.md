# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

## Getting Started
```bash
# Record a dataset
python -m alignit.record

# Use the dataset to train a model
python -m alignit.train

# Visualize dataset
python -m alignit.visualize --dataset.path=./data/duck

# Run inference
python -m alignit.infere
```


## Development

```bash
# Install the package in editable mode
git clone https://github.com/SpesRobotics/alignit.git
cd alignit
pip install -e .
```
