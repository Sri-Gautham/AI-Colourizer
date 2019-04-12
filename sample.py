import numpy as np
import pandas as pd
import os
import random
import keras
from PIL import Image
import csv
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LinearRegression


train_data = []
with open('input.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        train_data.append(row)
w, h = 281, 174
train_data = np.array(train_data).astype(int)
train_data = train_data / 255
print(np.array(train_data).shape)

output_data = []
with open('color.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        output_data.append(row)
y_train = np.array(output_data).astype(int)
y_train = y_train / 255
print(np.array(y_train).shape)

test_data = []
with open('data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        test_data.append(row)
w, h = 281, 174
w1, h1 = 641, 361
test_data = np.array(test_data).astype(int)
test_data = test_data / 255
print(np.array(test_data).shape)


lm = LinearRegression()
lm.fit(train_data,y_train)

predictions = lm.predict(test_data)

final1 = np.array(predictions).reshape(h1,w1,3)
final1 *=  255
final1 = final1.astype(np.uint8)
print(final1.shape)
#print(final.shape)
img = Image.fromarray(final1, 'RGB')
img.save('final1.png')
img.show()
