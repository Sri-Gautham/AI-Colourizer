import keras
from PIL import Image
import csv
from keras.datasets import mnist
from keras import losses
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

train_data = []
with open('input.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        train_data.append(row)
w, h = 281, 174
train_data = np.array(train_data).astype(int)
train_data = train_data / 255
#print(train_data)

test_data = []
with open('data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        test_data.append(row)
w1, h1 = 641, 361
test_data = np.array(test_data).astype(int)
#image_data = np.array(test_data).reshape(h1,w1).astype(np.uint8)
#img = Image.fromarray(image_data, mode = 'L')
#img.save('data.png')
#img.show()
test_data = test_data / 255
print(np.array(test_data).shape)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

output = loaded_model.predict(test_data)
#print(output)
#print(np.array(output).shape)
final1 = np.array(output).reshape(h1,w1,3)
final1 *= 255
final1 = final1.astype(np.uint8)
#print(final)
#print(final.shape)
img = Image.fromarray(final1, 'RGB')
img.save('final1.png')
img.show()