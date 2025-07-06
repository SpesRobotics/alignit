# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

## Getting Started
```bash
# Record a dataset
python3 -m alignit.record

# Use the dataset to train a model
python3 -m alignit.train

# Visualize dataset
python3 -m alignit.visualize

# Run inference
python3 -m alignit.infere
```


## Development

```bash
# Install the package in editable mode
git clone https://github.com/SpesRobotics/alignit.git
cd alignit
pip3 install -e .
```
