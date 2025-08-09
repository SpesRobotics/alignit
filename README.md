# AlignIt

Model-free real-time robot arm alignment using one or more RGB(D) cameras.

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="flex: 1; min-width: 180px">
    <img src="./media/record.gif" />
    <p style="text-align: center"><b>Record an Object</b> <br>data is being automatically collected and labeled</p>
  </div>
  <div style="flex: 1; min-width: 180px">
    <img src="./media/train.gif">
    <p style="text-align: center"><b>Train the Model</b></p>
  </div>
  <div style="flex: 1; min-width: 180px">
    <img src="./media/inference.gif" />
    <p style="text-align: center"><b>Align It</b> <br>model outputs relative poses to align the gripper with the object</p> 
  </div>
</div>

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
