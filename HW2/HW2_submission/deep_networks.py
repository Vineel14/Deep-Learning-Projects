import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow import keras

def deep_network_basic(in_shape, n_hidden, out_shape, metrics, args):
    
    # Build a sequential model
    model = Sequential()
    model.add(InputLayer(input_shape = (in_shape, )))
    for n in n_hidden:

        # Adding regularization term 
        if args.L1_regularization is not None:
            model.add(Dense(n, use_bias = True,kernel_regularizer = keras.regularizers.L1(args.L1_regularization), activation = args.activation_hidden))
            if args.dropout is not None:
                model.add(Dropout(args.dropout))
        else:
            model.add(Dense(n, use_bias = True, activation = args.activation_hidden))
            if args.dropout is not None:
                model.add(Dropout(args.dropout))
            
    model.add(Dense(out_shape,  use_bias = True, activation = args.activation_out))
    
    # Add optimizer to the model 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate)
    
    # Compile the model with metrics given
    model.compile(loss = 'mse', metrics = metrics, optimizer = opt)

    return model
