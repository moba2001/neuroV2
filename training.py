# %%
#%matplotlib inline
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

rng = np.random.RandomState(9)
# %%

input_width = 320
input_height = 120 
n_hid = 1000

# %%

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

with nengo.Simulator(model) as sim:
    print_error(sim)
    with open("nengo.pkl", 'wb') as outp:
        pickle.dump({"sim": sim, "a": a, "conn": conn}, outp, pickle.HIGHEST_PROTOCOL)
    with open("loopup.pkl", "wb") as outp:
        pickle.dump(lookup, outp, pickle.HIGHEST_PROTOCOL)
    with open("reverseloopup.pkl", "wb") as outp:
        pickle.dump(reverselookup, outp, pickle.HIGHEST_PROTOCOL)


# %%

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify(sim, frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            res = get_outs(sim, frame)
            idx = np.argmax(res)
            gesture = reverselookup[idx]
            gesture = "".join(gesture.split("_")[1:])
            print(f"Gesture: {gesture}, confidence: {softmax(res)[idx]}", end="\r")

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	#0 their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#if 1 == 1:
#    sim = ""
with nengo.Simulator(model) as sim:
    print("Sim ready...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

    # Create a queue to hold frames for classification
    frame_queue = queue.Queue()

    # Start the classification thread
    threading.Thread(target=classify, args=(sim, frame_queue,), daemon=True).start()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read(cv2.IMREAD_GRAYSCALE)

        if ret:
            frame = frame[25:145]
            #frame = 255 - frame
            frame[frame < 80] = 0 
            frame = adjust_gamma(frame, 3.5)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('Webcam Live', gray_frame)

            # Add the frame to the queue for classification
            if frame_queue.empty():  # Only keep the latest frame
                frame_queue.put(gray_frame.flatten())

            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
# %%


with open("nengo.pkl", 'rb') as outp:
    sim = pickle.load(outp)
    print(sim.model)
# %%
with nengo.Network(seed=0) as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    nengo_dl.configure_settings(stateful=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node(np.zeros(input_width * input_height))

    # add the first convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
        inp, shape_in=(input_width, input_height, 1)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # add the second convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
        x, shape_in=(input_width - 2, input_height - 2, 32)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # add the third convolutional layer
    x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
        x, shape_in=((input_width - 2) // 2 - 1, (input_height - 2) // 2 - 1, 64)
    )
    x = nengo_dl.Layer(neuron_type)(x)

    # linear readout
    out = nengo_dl.Layer(tf.keras.layers.Dense(units=10))(x)

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# %%

minibatch_size = 1
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)
# %%

tf_x_train = np.expand_dims(x_train, 0)
tf_y_train = np.expand_dims(y_train, 0).argmax(axis=1)

n_steps = 30
tf_x_test = np.tile(np.expand_dims(x_test, 0), (1, n_steps, 1))
tf_y_test = np.tile(np.expand_dims(y_test, 0), (1, n_steps, 1))
tf_y_test.shape
# %%

def classification_accuracy(y_true, y_pred):
    print("y_true", y_true.shape)
    print("y_pred", y_pred.shape)
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


# note that we use `out_p_filt` when testing (to reduce the spike noise)
sim.compile(loss={out_p_filt: classification_accuracy})
print(
    "Accuracy before training:",
    sim.evaluate(tf_x_test, {out_p_filt: tf_y_test}, verbose=0)["loss"],
)

# %%

do_training = False
if do_training:
    # run training
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
    )
    sim.fit(x_train, {out_p: y_train}, epochs=10)

    # save the parameters to file
    sim.save_params("./mnist_params")
else:
    # load parameters
    sim.load_params("./mnist_params")