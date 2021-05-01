import os, logging 

# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory
from flask_login         import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from jinja2              import TemplateNotFound

# App modules
from app        import app, lm, db, bc
from app.models import User 
from app.forms  import LoginForm, RegisterForm
import pickle
# import the necessary packages
from tensorflow.keras.models import load_model
# from google.colab.patches import cv2_imshow
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os

QUEUESIZE = 128
data_path = 'app\Training'
prediction_path = 'app\Predictions'
weight_file = os.path.join(data_path, 'forehand_activity-test-350-FINAL.model')
label_binerizer = os.path.join(data_path, 'forehand_lb-test-350-FINAL.pickle')
video_path = os.path.join(prediction_path, 'video', 'output.avi')

with open('app\Predictions\stroke.pickle', 'rb') as fp:
    item = pickle.load(fp)

def rolling_prediction():
    print("========================IN========================", item)

    step_model = load_model(weight_file)
    lb = pickle.loads(open(label_binerizer, 'rb').read())

    # mean initialization 
    mean = np.array([123.68, 116.779, 103.939][::1], dtype='float32')
    queue = deque(maxlen=QUEUESIZE)

        # capturing the video
    vs = cv2.VideoCapture(item)
    
    video_writer = None
    (W, H) = (None, None)

    # loop over each frame in the clip
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (W, H) = frame.shape[:2]
        output = frame.copy()

        # preprocessing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,  (224,224)).astype('float32')
        frame -= mean

        # giving out predictions and add to the queue
        preds = step_model.predict(np.expand_dims(frame, axis=0))[0]
        queue.append(preds)

            # perform prediction averaging over the current history of previous predictions
        results = np.array(queue).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]

        text = "activity: {}".format(label)
        cv2.putText(output, text, (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 3)

        if video_writer is None:
            # initialize our video video_writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0,(int(vs.get(3)),int(vs.get(4))))
            # write the output frame to disk
        video_writer.write(output)
        # print(output.shape)
            # display frames

        output = cv2.resize(output, (360, 240))
        cv2.imshow("output",output)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    print("[INFO] cleaning up...")
    video_writer.release()
    vs.release()

    return item