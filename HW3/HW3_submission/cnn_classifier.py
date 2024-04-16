# Required packages: tensorflow, keras, pandas, numpy, matplotlib, sklearn
import tensorflow as tf 
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


# Model building packages
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.layers import GlobalMaxPooling2D
from keras.layers import SpatialDropout2D
from keras.layers import InputLayer

# Create parser
args = argparse.ArgumentParser()

# Model Building function with default parameters
def create_cnn_classifier_network(
    image_size,
    nchannels,
    conv_layers = None,
    dense_layers = None,
    p_dropout = None,
    p_spatial_dropout = None,
    lambda_l2 = None,
    lambda_l1 = None,
    lrate=0.001,
    n_classes = 10,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    padding = 'same',
    flatten = True,
    args = args):

    '''
    Update kernel with regularization if specified to incorporate it into the model architecture.
    
    If either lambda_l2 or lambda_l1 regularization is provided, the kernel is updated accordingly.
    If both lambda_l2 and lambda_l1 are provided, l1_l2 regularization is applied.
    If only one type of regularization is provided, the corresponding regularization is applied.
    If no regularization is provided, the kernel is set to None.
    '''
    # Check if regularization is provided, then update kernel accordingly
    if (lambda_l2 is not None) or (lambda_l1 is not None):
        
        # Update kernel with l1_l2 regularization if both lambda_l2 and lambda_l1 are provided
        if (lambda_l2 is not None) and (lambda_l1 is not None):
            kernel = tf.keras.regularizers.l1_l2(lambda_l1, lambda_l2)
        else:
            # Update kernel with l1 regularization if only lambda_l1 is provided
            if lambda_l1 is not None:
                kernel = tf.keras.regularizers.l1(lambda_l1)
            
            # Update kernel with l2 regularization if only lambda_l2 is provided
            if lambda_l2 is not None:
                kernel = tf.keras.regularizers.l2(lambda_l2)
    # Set kernel to None if no regularization is provided
    else:
        kernel = None


    '''
    Model building section.
    
    1. Convolutional Layers:
    Input layer is added with image size and number of channels.
    Spatial dropout is applied if specified.
    Convolutional layers are added based on the provided configurations.
    Max Pooling is applied after each convolutional layer.
    Global Max Pooling Layer is added to identify significant features.
    
    2. Dense Layers:
    Dense layers are added based on the provided configurations.
    Dropout is applied if specified.
    Regularization (if any) is applied to the kernel prior to this step.
    '''
    # Create model
    model = tf.keras.Sequential()
    
    # Add Input Layer with image size and number of channels, also add Spatial Dropout if specified
    model.add(InputLayer(input_shape=(image_size[0],image_size[1], nchannels)))
    if p_spatial_dropout is not None: 
        model.add(SpatialDropout2D(p_spatial_dropout))
    
    # Add additional convolutional layers based on conv_layers configuration
    for i, n in enumerate(conv_layers):
        model.add(Conv2D(n['filters'], n['kernel_size'],padding = padding, activation = 'elu', name = 'conv_{}'.format(i)))
        
        # Add Spatial Dropout if specified
        if p_spatial_dropout is not None: 
            model.add(SpatialDropout2D(p_spatial_dropout))
            
        # Add Max Pooling if specified
        if n['pool_size'] is not None and n['pool_size'][0] > 1:    
            model.add(MaxPooling2D(pool_size = n['pool_size'], strides = n['strides'], name = 'pool_{}'.format(i+1)))
    
    # Global Max Pooling Layer to identify significant features
    model.add(GlobalMaxPooling2D())
    
    # Add dense layers
    for i,n in enumerate(dense_layers):
        
        # Add dense layer with kernel regularization
        model.add(Dense(n['units'], activation = 'elu', kernel_regularizer = kernel, name = 'dense_{}'.format(i+1)))
        
        # Add dropout if specified
        if p_dropout is not None:
            model.add(Dropout(p_dropout))

    # Add output layer with softmax activation function for classification
    model.add(Dense(n_classes, activation = 'softmax', name = 'output'))

    '''
    Add optimizer to the model and compile the model.
    
    Adam optimizer is used with a specified learning rate (from arguments or default value).
    Other parameters of the optimizer are kept default.
    For model compilation, loss and metrics are taken from arguments or default values.
    '''

    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
                  )

    # Return model
    return model

