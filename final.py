import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import os
from numpy import argmax
import matplotlib
img_rows=224
img_cols=224
num_channel=3

# Convert image compatible to classifier
import numpy as np
input_img = cv2.imread(r"C:\Users\dell\Desktop\Mini Project\TestIndivisual\3.jpeg") # Give the path of the image from 'TestIndivusal folder', that you want to classify
plt.imshow(input_img,cmap=matplotlib.cm.binary, interpolation="nearest")
input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
plt.imshow(input_img_resize,cmap=matplotlib.cm.binary, interpolation="nearest")
img_data = np.array([input_img_resize])
img_data = img_data.astype('float32')
img_data /= 255



from keras.models import load_model
import h5py
classes = ['Bulging Eyes','Cataracts','Crossed Eyes','Glucoma','Uveitis']
model = load_model('my_model.h5')
model.predict(img_data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
ynew = model.predict_classes(img_data)
print(classes[ynew[0]]) # Output 