
# coding: utf-8

# In[3]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *


# In[12]:


FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# In[19]:


def img_to_encoding(image):
    #img1 = cv2.imread(image_path, 1)
    #img = img1[...,::-1]
    img = np.around(np.transpose(image, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    x_train = np.array(x_train)
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding[0]


# In[20]:


#encoding = img_to_encoding("gan_makeup_data_96/with/001_1_y.jpg")


# In[21]:


#print (encoding)
#print (len(encoding))

