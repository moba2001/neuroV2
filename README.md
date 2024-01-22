# Gesture Recognition using Nengo

## Installation

### Leap Motion Sensor
Follow the instructions [here](https://docs.ultraleap.com/linux/) to install the linux drivers.

### Dataset
Download the dataset from [here](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

Put the dataset in a folder called input, so that `<project_root>/leapgestrecog/leapGestRecog` contains the folders `00` to `09`.

### Required Libraries
Install requirements using 
```shell 
pip install -r requirements.txt
```

## Training the models

### Nengo

Simply run 
```shell
python training_nengo.py
```
to train and evaluate the nengo model.
Image size as well as the ensemble size can be set by setting the variables `input_width`, `input_height` as well as `n_hid`.

## Inference

First kill the `leapd` process to get access to the cameras:
```shell 
sudo killall leapd
```

Then run the UI using
```shell 
python ui.py
```

## Troubleshoot

When getting a timeout opening the camera, start leapd using
```shell 
sudo leapd
```
After a few seconds press `Ctrl+C` to stop the process again.
 Once it is stopped, try to run the UI again.