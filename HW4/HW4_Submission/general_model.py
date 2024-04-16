# required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


# Model building packages
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.layers import UpSampling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer
from keras import Input
from keras.layers import Concatenate
from keras import Model
from keras.models import Sequential

args = argparse.ArgumentParser()

def create_autoencoder_network(input_shape=(256, 256, 26),
                               n_filters=[10, 5],
                               kernelSize=[3, 3],
                               pool_size=[2, 2],
                               spatial_dropout=None,
                               padding='same',  # 'same' is commonly used for padding in U-Net
                               lrate=0.001,
                               activation_convolution='elu',
                               skip_connections=False,
                               output_shape=7,
                               kernel = None,
                               args = args
                               ):
    inputs = Input(shape=input_shape)
    skips = []
    x = inputs

    # Downsampling
    for i, n in enumerate(n_filters):
        x = Conv2D(n, kernelSize[i], padding=padding, activation=activation_convolution)(x)
        if spatial_dropout is not None:
            x = SpatialDropout2D(spatial_dropout)(x)
        if skip_connections:
            skips.append(x)
        if pool_size[i] > 1:
            x = MaxPooling2D(pool_size=(pool_size[i], pool_size[i]))(x)

    # Upsampling
    for i, n in reversed(list(enumerate(n_filters))):
        if pool_size[i] > 1:
            x = UpSampling2D(size=(pool_size[i], pool_size[i]), interpolation='nearest')(x)
        if skip_connections and len(skips) > 0:
            skip_output = skips.pop()  # Get the corresponding skip connection
            x = Concatenate()([x, skip_output])
        x = Conv2D(n, kernelSize[len(kernelSize) - (i + 1)], padding=padding, activation=activation_convolution)(x)
        if spatial_dropout is not None:
            x = SpatialDropout2D(spatial_dropout)(x)

    # Output layer
    outputs = Conv2D(output_shape, (1, 1), padding='same', activation='softmax')(x)

    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

