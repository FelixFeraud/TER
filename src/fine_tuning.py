import tensorflow as tf
import keras
import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import resnet, vgg19, NASNetLarge

from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, Dropout, Input, BatchNormalization, Flatten
from keras.constraints import maxnorm
from keras.regularizers import l2, l1
from keras import callbacks
from keras import backend as K

from keras import optimizers

data_path = 'drive/My Drive/TER/'

print('lol')

print(tf.__version__)