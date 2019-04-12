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

epochs = 100
learning_rate = 0.01
decay_rate = learning_rate / epochs
sgd = SGD(lr=learning_rate, nesterov=False)


#getting grayscale data for training form input.csv
#resolution of input file 281 X 174
train_data = []
with open('input.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        train_data.append(row)
w, h = 281, 174
train_data = np.array(train_data).astype(int)
train_data = train_data / 255
print(np.array(train_data).shape)

#code block to create and save the image of the train data
#train_data = train_data[:,4]
#image_data = np.array(train_data).reshape(h,w).astype(np.uint8)
#img = Image.fromarray(image_data, mode = 'L')
#img.save('input.png')
#img.show()


#getting coloured data for training to use as y-label
output_data = []
with open('color.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        output_data.append(row)
y_train = np.array(output_data).astype(int)
y_train = y_train / 255
# dividing by 255 makes the data values from 0 to 1
print(np.array(y_train).shape)


#code block for creating image
#image2_data = np.array(output_data).reshape(h,w,3).astype(np.uint8)
#img = Image.fromarray(image2_data, 'RGB')
#img.save('color.png')
#img.show()


# data = np.zeros((h, w, 3), dtype=np.uint8)
#img = Image.fromarray(data, 'RGB')
#img.save('color.png')
#img.show()



#code for model design
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=9))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.summary()


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, y_train, epochs=500, batch_size=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")