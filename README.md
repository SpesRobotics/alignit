# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.


| ![Data Recording](media/record.gif) | ![Model Training](media/train.gif) | ![Inference](media/inference.gif) |
|:------------------------------------:|:-----------------------------------:|:----------------------------------:|
| **Record an Object** <br> (data is being automatically collected and labeled) | **Train the Model**                  | **Align It** <br /> (model outputs relative poses to align the gripper with the object)                     |


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