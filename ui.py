from dataclasses import dataclass
import time
import cv2
import leapuvc
import psutil
import os
from nengoCNN import NengoCNNPredictor, preprocess_img_cnn_nengo

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread(timeout=60)
leap.start()
leap.setExposure(50)

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


@dataclass 
class ClassificationOutput:
    name = ""
    value = ""

    @property
    def out(self):
        return self.name + ": " + self.value

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

with open("reverseloopup_tf.pkl", "rb") as f:
    reverselookup_tf = pickle.load(f)

model = load_model('./my_model_small.h5')
model_big = load_model('./my_model_medium.h5')

model_nengo_cnn = NengoCNNPredictor(params_file="keras_to_snn_params", input_height=28, input_width=28)
model_nengo_cnn_big = NengoCNNPredictor(params_file="keras_to_snn_big_params")


tiled_test_images = preprocess_img_cnn_nengo(leftRightImage[0], width=28, height=28)
tiled_test_images_big = preprocess_img_cnn_nengo(leftRightImage[0])

model_nengo_cnn.predict(tiled_test_images) # Load model
model_nengo_cnn_big.predict(tiled_test_images_big) # Load model

def convert_img(img, width=320, height =120) :
    input_width = width
    input_height = height
    
    img = img.reshape((1, input_height, input_width, 1))

    return img

def get_outs(simulator, images):
    # encode the images to get the ensemble activations
    _, acts = nengo.utils.ensemble.tuning_curves(a, simulator, inputs=images)

    # decode the ensemble activities using the connection's decoders
    return np.dot(acts, simulator.data[conn].weights.T)

def get_time_difference(before_threads, after_threads, thread_id):
    '''Calculate the time difference of two threads. Search for correct thread only now to reduce overhead'''
    for thread in before_threads:
        if thread.id == thread_id:
            before = thread.user_time + thread.system_time
    
    for thread in after_threads:
        if thread.id == thread_id:
            after = thread.user_time + thread.system_time
    
    return round((after - before) * 1000)

def classify_tf(my_model, frame_queue, out: ClassificationOutput):
    global latest_tf_output, running
    process, tid = psutil.Process(os.getpid()), threading.get_native_id()
    print(f"Classifying TF on PID {os.getpid()} Thread ID {tid}")
    total_time = 0
    total_classifications = 0
    while running:
        sleep(0.01)
        if not frame_queue.empty():
            try:
                start = datetime.now()
                total_classifications += 1
                before_threads = process.threads()
                frame = frame_queue.get()
                prediction = my_model.predict(frame, verbose=None)
                idx = np.argmax(prediction)
                after_threads = process.threads()
                end = datetime.now()
                gesture = reverselookup_tf[idx]
                gesture = "".join(gesture.split("_")[1:])
                diff = get_time_difference(before_threads, after_threads, tid)
                total_time += diff
                out.value = f"Gesture: {gesture} ({idx}), Confidence: {round(softmax(prediction[0][idx]), 3)}, CPU time: {diff} ms, avg {round(total_time / total_classifications)} ms, Real Time: {round((end-start).total_seconds() * 1000)} ms"
            except psutil.NoSuchProcess:
                """Sometimes psutils failes to find the process info"""
                pass



def classify_nengo(sim, frame_queue, out: ClassificationOutput):
    global latest_nengo_output, running
    process, tid = psutil.Process(os.getpid()), threading.get_native_id()
    print(f"Classifying Nengo on PID {os.getpid()} Thread ID {tid}")
    total_time = 0
    total_classifications = 0
    while running:
        sleep(0.01)
        if not frame_queue.empty():
            try:
                total_classifications += 1
                start = datetime.now()
                before_threads = process.threads()
                frame = frame_queue.get()
                res = get_outs(sim, frame)
                idx = np.argmax(res)
                after_threads = process.threads()
                end = datetime.now()
                gesture = reverselookup[idx]
                gesture = "".join(gesture.split("_")[1:])
                diff = get_time_difference(before_threads, after_threads, tid)
                total_time += diff
                out.value = f"Gesture: {gesture}({idx}), Confidence: {round(softmax(res)[idx], 3)}, Time: {diff} ms, avg {round(total_time / total_classifications)} ms, Real Time: {round((end-start).total_seconds() * 1000)} ms"
            except psutil.NoSuchProcess:
                """Sometimes psutils failes to find the process info"""
                pass

def show_frame():
    #_, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    newFrame, leftRightImage = leap.read()
    frame = leftRightImage[0]
    if (newFrame):
        if frame_queue.empty():
            gray_frame = cv2.resize(frame, (input_width ,input_height), interpolation= cv2.INTER_LINEAR)
            gray_frame = gray_frame / 255 * 2 - 1
            frame_queue.put(gray_frame.flatten())

        if frame_queue2.empty():
            gray_frame = cv2.resize(frame, (28, 28), interpolation= cv2.INTER_LINEAR)
            gray_frame = gray_frame / 255 * 2 - 1
            gray_frame = convert_img(gray_frame, width=28, height=28)
            frame_queue2.put(gray_frame)

        if frame_queue3.empty():
            gray_frame = cv2.resize(frame, (160 ,60), interpolation= cv2.INTER_LINEAR)
            gray_frame = gray_frame / 255 * 2 - 1
            gray_frame = convert_img(gray_frame, width=160, height=60)
            frame_queue3.put(gray_frame)

        
        if frame_queue5.empty():
            gray_frame = preprocess_img_cnn_nengo(frame, width=28, height=28)
            frame_queue5.put(gray_frame)

        if frame_queue6.empty():
            gray_frame = preprocess_img_cnn_nengo(frame)
            frame_queue6.put(gray_frame)

        frame = cv2.resize(frame, preview_shape, interpolation= cv2.INTER_LINEAR)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        text_label1.config(text=o1.out)
        text_label2.config(text=o2.out)
        text_label3.config(text=o3.out)
        text_label5.config(text=o5.out)
        text_label6.config(text=o6.out)

    lmain.after(10, show_frame)

def close():
    global running
    running = False
    print("Waiting for thread 1 to finish")
    t1.join()
    print("Waiting for thread 2 to finish")
    t2.join()
    t3.join()
    t5.join()
    t6.join()
    leap.timeout = 0
    time.sleep(0.4)
    leap.cam.release()
    window.destroy()
    print("Exit")

frame_queue = queue.Queue()
frame_queue2 = queue.Queue()
frame_queue3 = queue.Queue()
frame_queue5 = queue.Queue()
frame_queue6 = queue.Queue()

running=True

o1 = ClassificationOutput()
o1.name = "Nengo"

o2 = ClassificationOutput()
o2.name = "CNN"

o3 = ClassificationOutput()
o3.name = "CNN Big"

o5 = ClassificationOutput()
o5.name = "Nengo CNN"

o6 = ClassificationOutput()
o6.name = "Nengo CNN Big"

t1 = threading.Thread(target=classify_nengo, args=(sim, frame_queue, o1), daemon=True)
t2 = threading.Thread(target=classify_tf, args=(model, frame_queue2, o2), daemon=True)
t3 = threading.Thread(target=classify_tf, args=(model_big, frame_queue3, o3), daemon=True)
t5 = threading.Thread(target=classify_tf, args=(model_nengo_cnn, frame_queue5, o5), daemon=True)
t6 = threading.Thread(target=classify_tf, args=(model_nengo_cnn_big, frame_queue6, o6), daemon=True)

t1.start()
t2.start()
t3.start()
t5.start()
t6.start()


window = tk.Tk()
window.title("Webcam Preview")

input_width = 320
input_height = 120 

preview_shape = (1920, 1080)

#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

lmain = tk.Label(window)
lmain.pack()

text_label1 = tk.Label(window, text="Your Text Here", anchor='w')
text_label1.pack(fill="x")

text_label2 = tk.Label(window, text="Your Text Here", anchor='w')
text_label2.pack(fill="x")

text_label3 = tk.Label(window, text="Your Text Here", anchor='w')
text_label3.pack(fill="x")

text_label5 = tk.Label(window, text="Your Text Here", anchor='w')
text_label5.pack(fill="x")

text_label6 = tk.Label(window, text="Your Text Here", anchor='w')
text_label6.pack(fill="x")

show_frame_thread = threading.Thread(target=show_frame)
show_frame_thread.start()

window.protocol("WM_DELETE_WINDOW", close)
window.mainloop()