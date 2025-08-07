# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

## Getting Started

AlignIt uses [draccus](https://github.com/dlwh/draccus) for configuration management. You can pass parameters via command line or YAML files.

```bash
# Record a dataset with default parameters
python -m alignit.record

# Record with custom parameters
python -m alignit.record --episodes=50 --trajectory.z_step=0.001

# Record using a config file
python -m alignit.record --config_path=configs/record.yaml

# Train a model
python -m alignit.train --config_path=configs/train.yaml

# Train with custom learning rate
python -m alignit.train --config_path=configs/train.yaml --learning_rate=0.0005

# Visualize dataset
python -m alignit.visualize --config_path=configs/visualize.yaml

# Run inference/alignment
python -m alignit.infere --config_path=configs/infer.yaml

# Run inference with custom convergence tolerances
python -m alignit.infere --lin_tolerance=0.001 --ang_tolerance=1.0 --max_iterations=1000
```

### Configuration Files

Example configurations are provided in the `configs/` directory:
- `configs/record.yaml` - Dataset recording parameters
- `configs/train.yaml` - Model training parameters  
- `configs/infer.yaml` - Inference/alignment parameters
- `configs/visualize.yaml` - Dataset visualization parameters

You can override any parameter from the command line even when using a config file.


## Development

```bash
# Install the package in editable mode
git clone https://github.com/SpesRobotics/alignit.git
cd alignit
pip install -e .
```
