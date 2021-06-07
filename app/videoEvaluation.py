
# Python modules
import os, logging 
# from numba import jit, cuda

# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory
from flask_login         import login_user, logout_user, current_user, login_required
from werkzeug.exceptions import HTTPException, NotFound, abort
from jinja2              import TemplateNotFound

# App modules
from app        import app, lm, db, bc
from app.models import User 
from app.forms  import LoginForm, RegisterForm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.vgg19 import VGG19
import cv2
import math
import os
from glob import glob
from scipy import stats as s
from keras.optimizers import Adam

# input_path = os.path.join(data_path, 'static\uploads\543319.jpg')
data_path = 'app\Training'
weight_file = os.path.join(data_path, 'weights_strokes_V1_5_epochs-250.hdf5')
train_csv = os.path.join(data_path, 'train_v1_5_1.csv')
temp = os.path.join(data_path, 'temp')

initial_lr = 0.001

# base model
base_model = VGG19(weights='imagenet',include_top=False)

def build_model():
    
    # layers
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(25088,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # load weight file
    model.load_weights(weight_file)
    model.compile(optimizer=Adam(learning_rate=initial_lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# @jit(target ="cuda")
def stroke_evaluation(video_file):
    
    model = build_model()
    
    train = pd.read_csv(train_csv)
    y = train['Class']
    y = pd.get_dummies(y)

    predict = []

    # extract frames from each video
    count = 0
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    x = 1
    # remove other files from temp
    files = glob(temp + '\\*')
    for f in files:
        os.remove(f)

    while (cap.isOpened()):
        frame_d = cap.get(1)
        ret, frame = cap.read()

        if (ret != True):
            break
            
        filename = temp + '\\' + "_frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)

    cap.release()

    # reading all the frames from temp folder
    images = glob(temp + '\\*'+".jpg")
    prediction_images = []
    prediction_images.clear()
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)

    prediction_images = np.array(prediction_images)
    prediction_images = base_model.predict(prediction_images)
    prediction_images = prediction_images.reshape(prediction_images.shape[0] ,7*7*512)
    prediction = np.argmax(model.predict(prediction_images), axis=-1)
    predict.append(y.columns.values[s.mode(prediction)[0][0]])

    # print(np.argmax(model.predict(prediction_images), axis=-1))
    # print(model.predict(prediction_images))

    return predict

