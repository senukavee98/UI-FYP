
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
