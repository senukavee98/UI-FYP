
# Python modules
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

# from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Reshape, Input
# from keras.models import Model
# from keras.layers import Conv2D, MaxPooling2D
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.inception_v3 import InceptionV3
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg19 import VGG19
# import cv2
# import math
# import os
# from glob import glob
from scipy import stats as s
# from keras.optimizers import Adam



@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    print(request.files, '<<<< FILE DATA >>>')
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join('app/static/uploads', uploaded_file.filename))

    # VGG19 base model
    # base_model = VGG19(weights='imagenet',include_top=False)
    
    return redirect('/')