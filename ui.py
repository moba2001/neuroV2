import cv2
import leapuvc


# Start the Leap Capture Thread
leap = leapuvc.leapImageThread(timeout=60)
leap.start()

while(leap.running):
    newFrame, leftRightImage = leap.read()
    if newFrame:
        break

if leftRightImage is None:
    exit(0)

import nengo
import numpy as np
import threading
from datetime import datetime
import os
from datetime import datetime
import pickle
import queue
from time import sleep
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

now = datetime.now()
print("Loading nenog model...")
with open("nengo.pkl", 'rb') as outp:
    n = pickle.load(outp)
    sim = n["sim"]
    a = n["a"]
    conn = n["conn"]

print(f"Loading nengo model took {datetime.now() - now}")

with open("reverseloopup.pkl", "rb") as f:
    reverselookup = pickle.load(f)

loaded_model = load_model('./my_model.h5')

def convert_img(img, width=320, height =120) :
    input_width = width
    input_height = height

    img = img.resize((input_width, input_height))
    img = np.array(img, dtype = 'float32')
    img = img.reshape((1, input_height, input_width, 1))

    return img

def get_outs(simulator, images):
    # encode the images to get the ensemble activations
    _, acts = nengo.utils.ensemble.tuning_curves(a, simulator, inputs=images)

    # decode the ensemble activities using the connection's decoders
    return np.dot(acts, simulator.data[conn].weights.T)

latest_nengo_output = ""
latest_tf_output = ""

def classify_tf(my_model, frame_queue):
    global latest_tf_output
    while True:
        sleep(0.01)
        if not frame_queue.empty():
            before = datetime.now()
            frame = frame_queue.get()
            frame = convert_img(frame)
            prediction = my_model.predict(frame)
            idx = np.argmax(prediction)
            diff = datetime.now() - before
            gesture = reverselookup[idx]
            gesture = "".join(gesture.split("_")[1:])
            print(gesture)
            latest_tf_output = f"Gesture: {gesture}, Confidence: {round(softmax(prediction)[idx], 3)}, Time: {diff.microseconds / 1000}"



def classify_nengo(sim, frame_queue):
    global latest_nengo_output
    while True:
        sleep(0.01)
        if not frame_queue.empty():
            before = datetime.now()
            frame = frame_queue.get()
            res = get_outs(sim, frame)
            idx = np.argmax(res)
            diff = datetime.now() - before
            gesture = reverselookup[idx]
            gesture = "".join(gesture.split("_")[1:])
            latest_nengo_output = f"Gesture: {gesture}, Confidence: {round(softmax(res)[idx], 3)}, Time: {diff.microseconds / 1000}"

def show_frame():
    #_, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    newFrame, leftRightImage = leap.read()
    frame = leftRightImage[0]
    if (newFrame):
        if frame_queue.empty():
            gray_frame = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (input_width ,input_height), interpolation= cv2.INTER_LINEAR)
            gray_frame = gray_frame / 255 * 2 - 1
            frame_queue.put(gray_frame.flatten())

        if frame_queue2.empty():
            gray_frame = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (input_width ,input_height), interpolation= cv2.INTER_LINEAR)
            gray_frame = gray_frame / 255 * 2 - 1
            frame_queue2.put(gray_frame.flatten())

        frame = cv2.resize(frame, preview_shape, interpolation= cv2.INTER_LINEAR)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        text_label1.config(text=latest_nengo_output)
        text_label2.config(text=latest_tf_output)
    lmain.after(10, show_frame)

def close():
    cap.release()
    window.destroy()

frame_queue = queue.Queue()
frame_queue2 = queue.Queue()
threading.Thread(target=classify_nengo, args=(sim, frame_queue,), daemon=True).start()
threading.Thread(target=classify_tf, args=(sim, frame_queue2,), daemon=True).start()

window = tk.Tk()
window.title("Webcam Preview")

input_width = 320
input_height = 120 

preview_shape = (1080, 720)

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

lmain = tk.Label(window)
lmain.pack()

text_label1 = tk.Label(window, text="Your Text Here", anchor='w')
text_label1.pack(fill="x")

text_label2 = tk.Label(window, text="Your Text Here", anchor='w')
text_label2.pack(fill="x")

show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

window.protocol("WM_DELETE_WINDOW", close)
window.mainloop()