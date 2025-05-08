# GestureAmp

GestureAmp is a tool to:

1) Train a object detection model (a Keras YOLOv8 implementation) based on the Hagrid dataset.
2) Run a camera feed to identify hand gestures.
3) Control a Media Player Daemon out of the detected gestures

So, you can control playback right from your couch without the need to touch any phone! :D

The tool is not meant to be ditributed or used out-of-the-box but rather to keep track of the progress in the project. Nevertheless I do provide some rough installation / usage instructions.

## Installation

1. clone this repo
```
git clone git@github.com:pabloportela/GestureAmp.git
cd GestureAmp
```

## Train an object detector
1. Download Hagrid dataset from https://github.com/hukenovs/hagrid/tree/master
2. create a virtual environment, install the requirements to train a model
```
python -m venv .venv_train
. .venv_train/bin/activate
pip install -r requirements_train.txt
```
3. Edit `dataset.py` and `common.py` to reflect the location of the dataset and the gsetures you want to include.
4. `python3 train.py`
