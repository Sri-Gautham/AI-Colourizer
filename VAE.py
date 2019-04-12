import numpy as np
import pandas as pd
import os
import random
import keras
from PIL import Image
from keras.models import model_from_json
import csv
import statsmodels.api as sm
import sklearn
import numpy as np
from numpy import genfromtxt
ip = genfromtxt('input.csv', delimiter=',')
color = genfromtxt('color.csv', delimiter=',')
test_data = genfromtxt('data.csv', delimiter=',')
ip/=255
color/=255
test_data/=255
data_size = ip.shape[0]
val_split = 0.2
val_size = round(val_split*data_size)
val_ix = np.random.choice(data_size, val_size, replace=False)
train_ix = np.array(list(set(range(data_size))-set(val_ix)))
print('Train dset size - {} \nTest dset size - {}'.format(train_ix.size, val_ix.size))
x_val, y_val = ip[val_ix[:-3]], color[val_ix[:-3]]
train_ix = np.append(train_ix, val_ix[-1])
x_train, y_train = ip[train_ix], color[train_ix]
x_test = test_data
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K

batch_size = 4
latent_dim = 512
x = Input(batch_shape=(None, 9))
h = Dense(64, activation='relu')(x)
h = Dense(256, activation='relu')(x)
h = Dropout(0.2)(h)
#h = Dense(latent_dim, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
epsilon_std = 1.0
def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = 4
    
    if(z_mean.shape[0]!=64):
        print(z_mean.shape[0])
        batch_size = z_mean.shape[0]
        
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
decoder_h1 = Dense(256, activation='relu')
decoder_h2 = Dense(64, activation='relu')
decoder_mean = Dense(3, activation='sigmoid')
h_decoded = decoder_h1(z)
h_decoded = Dropout(0.2)(h_decoded)
h_decoded = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)
# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h2(decoder_h1(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
from keras import losses
def vae_loss(y_train, x_decoded_mean):
    xent_loss = losses.binary_crossentropy(y_train, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=['accuracy'])

vae.fit(x_train, y_train,
        shuffle=True,
        epochs=100,
        batch_size=batch_size,
        validation_data=(x_val, y_val))

# serialize model to JSON
model_json = vae.to_json()
with open("vae.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("vae.h5")
print("Saved model to disk")