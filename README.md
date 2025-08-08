# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

### Getting Started

```bash
# Record a dataset
python -m alignit.record --dataset.path=./data/test

# Use the dataset to train a model
python -m alignit.train --dataset.path=./data/test --model.path=./data/test_model.pth

# Visualize dataset
python -m alignit.visualize --dataset.path=./data/test

# Run inference
python -m alignit.infere --model.path=./data/test_model.pth
```

### Development

```bash
# Install the package in editable mode
git clone https://github.com/SpesRobotics/alignit.git
cd alignit
pip install -e .

# Run tests
python -m pytest
```

### HuggingFace Hub Integration

```bash
# Record and push to HuggingFace Hub
python -m alignit.record --dataset.hf_username=username --dataset.hf_dataset_name=dataset-name

# Train using HuggingFace dataset
python -m alignit.train --dataset.hf_dataset_name=username/dataset-name --model.path=./data/test_model.pth

# Visualize HuggingFace dataset
python -m alignit.visualize --dataset.hf_dataset_name=username/dataset-name
```

### Weights & Biases Integration

```bash
# Train with wandb experiment tracking
python -m alignit.train --wandb_project=my-robot-project --wandb_run_name=experiment-1

# Combine HuggingFace Hub and wandb
python -m alignit.train --dataset.hf_dataset_name=username/dataset-name --wandb_project=my-robot-project
```