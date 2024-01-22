# %%
#%matplotlib inline
from datetime import datetime
import nengo
import numpy as np
import time

import tkinter as tk
import os
from PIL import Image
from sklearn.model_selection import train_test_split

from nengo_extras.data import one_hot_from_labels
from nengo_extras.matplotlib import tile
from nengo_extras.vision import Gabor, Mask
import cv2 
import threading
import queue
import pickle
import nengo_dl
import tensorflow as tf
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, fbeta_score

rng = np.random.RandomState(9)
# %%

input_width = 320
input_height = 120 
n_hid = 1000

# %%
# Create lookup tables for class id <=> name mappings
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('./input/leapgestrecog/leapGestRecog/00/'):
    if not j.startswith('.'): 
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup

# %%
# Load data

x_data = []
y_data = []
datacount = 0 
for i in range(0, 10):
    for j in os.listdir('./input/leapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir('./input/leapgestrecog/leapGestRecog/0' + 
                                str(i) + '/' + j + '/'):
                img = Image.open('./input/leapgestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((input_width, input_height))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count

x_data = np.array(x_data, dtype = 'float32')

# For Nengo, set values between 0 and 1
x_data = x_data / 255 * 2 - 1
x_data = x_data.reshape(datacount, -1)

y_data = np.array(y_data)
y_data = y_data.reshape(datacount)

# %%

y_data_1h = one_hot_from_labels(y_data)

# %%

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data_1h,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

# %%

# Setup nengo net
# https://www.nengo.ai/nengo-extras/examples/mnist_single_layer.html
n_vis = x_train.shape[1]
n_out = y_train.shape[1]

ens_params = dict(
    eval_points=x_train,
    neuron_type=nengo.LIFRate(),
    intercepts=nengo.dists.Choice([0.1]),
    max_rates=nengo.dists.Choice([100]),
)

solver = nengo.solvers.LstsqL2(reg=0.01)

with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_hid, n_vis, **ens_params)
    v = nengo.Node(size_in=n_out)
    conn = nengo.Connection(
        a, v, synapse=None, eval_points=x_train, function=y_train, solver=solver
    )
    print(conn)


def get_outs(simulator, images):
    # encode the images to get the ensemble activations
    _, acts = nengo.utils.ensemble.tuning_curves(a, simulator, inputs=images)

    # decode the ensemble activities using the connection's decoders
    return np.dot(acts, simulator.data[conn].weights.T)


def get_error(simulator, images, labels):
    # the classification for each example is index of
    # the output dimension with the highest value
    val = np.argmax(get_outs(simulator, images), axis=1) != np.argmax(labels, axis=1)
    return val


def print_error(simulator):
    train_error = 100 * get_error(simulator, x_train, y_train).mean()
    test_error = 100 * get_error(simulator, x_validate, y_validate).mean()
    print("Train/test error: %0.2f%%, %0.2f%%" % (train_error, test_error))



# %%
encoders = Gabor().generate(n_hid, (64, 64), rng=rng)
encoders = Mask((input_height, input_width)).populate(encoders, rng=rng, flatten=True)
a.encoders = encoders

tile(encoders.reshape((-1, input_height, input_width)), rows=4, cols=6, grid=True)

# %%

# Train, run eval and save models

t1 = datetime.now()
with nengo.Simulator(model) as sim:
    t2 = datetime.now()
    print(f"Training took {t2-t1} seconds")
    print_error(sim)
    y_pred = np.argmax(get_outs(sim, x_validate), axis=1)
    y_true = np.argmax(y_validate, axis=1)
    print(f"F2-Score: {fbeta_score(y_true, y_pred, beta=2, average='macro')}")
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index = [reverselookup[i].split("_")[1] for i in reverselookup.keys()],
                    columns = [reverselookup[i].split("_")[1] for i in reverselookup.keys()])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    with open(f"nengo_{n_hid}.pkl", 'wb') as outp:
        pickle.dump({"sim": sim, "a": a, "conn": conn}, outp, pickle.HIGHEST_PROTOCOL)
    with open("loopup.pkl", "wb") as outp:
        pickle.dump(lookup, outp, pickle.HIGHEST_PROTOCOL)
    with open("reverseloopup.pkl", "wb") as outp:
        pickle.dump(reverselookup, outp, pickle.HIGHEST_PROTOCOL)
# %%
