'''
Advanced Machine Learning, 2024
HW 3 Base Code

Author: Andrew H. Fagg (andrewhfagg@gmail.com)

Image classification for the Core 50 data set

Updates for using caching and GPUs
- Batch file:
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
or
#SBATCH --partition=disc_dual_a100_students
#SBATCH --cpus-per-task=64

- Command line options to include
--cache $LSCRATCH                              (use lscratch to cache the datasets to local fast disk)
--batch 4096                                   (this parameter is per GPU)
--gpu
--precache datasets_by_fold_4_objects          (use a 4-object pre-constructed dataset)

Notes: 
- batch is now a parameter per GPU.  If there are two GPUs, then this number is doubled internally.
   Note that you must do other things to make use of more than one GPU
- 4096 works on the a100 GPUs
- The prcached dataset is a serialized copy of a set of TF.Datasets (located on slow spinning disk).  
Each directory contains all of the images for a single data fold within a couple of files.  Loading 
these files is *a lot* less expensive than having to load the individual images and preprocess them 
at the beginning of a run.
- The cache is used to to store the loaded datasets onto fast, local SSD so they can be fetched quickly
for each training epoch

'''

import sys
import argparse
import pickle
import pandas as pd
import wandb
import socket

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model

#############
# REMOVE THESE LINES if symbiotic_metrics, job_control, networks are in the same directory
tf_tools = "../../../../tf_tools/"
sys.path.append(tf_tools + "experiment_control")
sys.path.append(tf_tools + "networks")
#############

# Provided
from symbiotic_metrics import *
from job_control import *
from core50 import *
from hw3_parser import *

# You need to provide this yourself
from cnn_classifier import *
import matplotlib.pyplot as plt

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    This is trivial right now

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    '''
    if args.exp_type is None:
        p = {'rotation': range(5)}
    elif args.exp_type == 'L2':
        p = {'rotation': range(5), 'L2_regularization': [1.0, 10.0]}
    else:
        assert False, "Unrecognized exp_type (%s)"%args.exp_type

    return p


#################################################################
def check_args(args):
    '''
    Check that the input arguments are rational

    '''
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.spatial_dropout is None or (args.spatial_dropout > 0.0 and args.dropout < 1)), "Spatial dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L1_regularization is None or (args.L1_regularization > 0.0 and args.L1_regularization < 1)), "L1_regularization must be between 0 and 1"
    assert (args.L2_regularization is None or (args.L2_regularization > 0.0 and args.L2_regularization < 1)), "L2_regularization must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if index is None:
        return ""
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''
    # Hidden unit configuration
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Conv configuration
    conv_size_str = '_'.join(str(x) for x in args.conv_size)
    conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)
    pool_str = '_'.join(str(x) for x in args.pool)
    
    # Dropout
    if args.dropout is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout)
        
    # Spatial Dropout
    if args.spatial_dropout is None:
        sdropout_str = ''
    else:
        sdropout_str = 'sdrop_%0.3f_'%(args.spatial_dropout)
        
    # L1 regularization
    if args.L1_regularization is None:
        regularizer_l1_str = ''
    else:
        regularizer_l1_str = 'L1_%0.6f_'%(args.L1_regularization)

    # L2 regularization
    if args.L2_regularization is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_regularization)


    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
        
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f_"%args.lrate
    
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/image_%s%sCsize_%s_Cfilters_%s_Pool_%s_Pad_%s_hidden_%s_%s%s%s%s%sntrain_%02d_rot_%02d"%(args.results_path,
                                                                                           experiment_type_str,
                                                                                           label_str,
                                                                                           conv_size_str,
                                                                                           conv_filter_str,
                                                                                           pool_str,
                                                                                           args.padding,
                                                                                           hidden_str, 
                                                                                           dropout_str,
                                                                                           sdropout_str,
                                                                                           regularizer_l1_str,
                                                                                           regularizer_l2_str,
                                                                                           lrate_str,
                                                                                           args.Ntraining,
                                                                                           args.rotation)

#################################################################
#
#  DO NOT USE THIS FOR HW 3
#

def load_data_set_by_folds(args, objects=[4,5,6,8], seed=42):
    '''
    Create the data set from scratch using the specified objects.

    The underlying caching is done by fold (as opposed to data set)

    The Core 50 data set has:
    - 10 object classes
    - 5 object instances per class
    - 11 background conditions in which each of the object instances are imaged in

    The specific arrangement of operations defines the particular problem that we are solving.
    Specifically:
    - 'example': one object instance of a class per fold (with all conditions)
    - 'condition': two conditions per fold (all object instances)

    :param args: Command line arguments
    :param objects: List of objects to include in the training/evaluation process (integers 0...9)
    :param seed: Random seed
    
    :return: TF Datasets for the training, validation and testing sets + number of classes
    '''

    # Test create object-based rotations
    core = Core50(args.dataset + '/' + args.meta_dataset)

    # Set the problem class IDs
    # Object 4->C0; Object 5->C1; Object 6->C2; Object 8->C3;
    #   ignore all other objects

    core.set_problem_class_by_equality('class', objects)

    # Select only these object classes (remove all others)
    core.filter_problem_class()

    # Create folds
    if args.problem == 'condition':
        # Folds by pairs of condition ((1,2), (3,4), ...)
        folds = core.create_subsets_by_membership('condition', list(zip(range(1,11,2),range(2,11,2))))
    elif args.problem == 'example':
        # Folds by examples
        print(list(range(5)))
        folds = core.create_subsets_by_membership('example', [[i] for i in range(5)])
    else:
        assert False, "Bad learning problem specified."

    # Shuffle the rows in each of the dataframes
    folds = [df.sample(frac=1, random_state=seed) for df in folds]

    # Check to make sure that argument matches that actual number of folds
    assert len(folds) == args.Nfolds, "args.Nfolds does not match actual number of folds"

    # Create the fold-wise data sets
    ds_folds = []
    for i, f in enumerate(folds):
        # Create the corresponding Datasets: only up to caching
        ds = core.create_dataset(f, args.dataset + '/core50_128x128',
                                 column_file='fname',
                                 column_label='problem_class',
                                 batch_size=1,
                                 prefetch=0,
                                 num_parallel_calls=args.num_parallel_calls,
                                 cache=args.cache,
                                 repeat=False,
                                 shuffle=0,
                                 dataset_name='core50_objects_10_fold_%d'%i,
                                 py_func=(not args.no_use_py_func))
        ds_folds.append(ds)


    # Create training/validation/test DFs
    # This step does the batching/prefetching/repeating/shuffling
    ds_training, ds_validation, ds_testing = core.create_training_validation_testing_from_datasets(args.rotation,
                                                                                                   ds_folds,
                                                                                                   args.Ntraining,
                                                                                                   batch_size=args.batch,
                                                                                                   prefetch=args.prefetch,
                                                                                                   repeat=args.repeat,
                                                                                                   shuffle=args.shuffle)

    # Done
    return ds_training, ds_validation, ds_testing, len(objects)



#################################################################
#
#  Use for HW 3: load pre-assembled Datasets
#

def load_precached_folds(args, seed=42):
    '''
    Load the data set from the arguments.

    The underlying files represent individual folds.  Several are combined together to
    create the training set.

    The Core 50 data set has:
    - 10 object classes
    - 5 object instances per class
    - 11 background conditions in which each of the object instances are imaged in
    
    :param args: Command line arguments
    :param seed: Random seed
    
    :return: TF Datasets for the training, validation and testing sets + number of classes
    '''

    # Open meta data file to pull out the number of objects
    fname = args.dataset + '/' + args.precache + '/meta.csv'
    print(fname)
    df = pd.read_csv(fname)
    nobjects = df['nclasses'][0]
    nfolds = df['nfolds'][0]

    print('Number of Objects:', nobjects)
    print('Number of Folds:', nfolds)

    # Test create object-based rotations
    core = Core50(args.dataset + '/' + args.meta_dataset)

    # Check to make sure that argument matches that actual number of folds
    assert nfolds == args.Nfolds, "args.Nfolds does not match actual number of folds"

    # Load the fold-wise data sets
    ds_folds = []
    for i in range(args.Nfolds):
        name = 'core50_objects_10_fold_%d'%i

        # Create the foldwise datasets
        ds = tf.data.Dataset.load('%s/%s/%s'%(args.dataset, args.precache, name))
        ds_folds.append(ds)


    # Create training/validation/test DFs
    # This step does the batching/prefetching/repeating/shuffling
    ds_training, ds_validation, ds_testing = core.create_training_validation_testing_from_datasets(args.rotation,
                                                                                                   ds_folds,
                                                                                                   args.Ntraining,
                                                                                                   cache=args.cache,
                                                                                                   batch_size=args.batch,
                                                                                                   prefetch=args.prefetch,
                                                                                                   repeat=args.repeat,
                                                                                                   shuffle=args.shuffle)
    # Done
    return ds_training, ds_validation, ds_testing, nobjects
    
#################################################################
def execute_exp(args=None, multi_gpus=False):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    :param multi_gpus: True if there are more than one GPU
    '''

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch*multi_gpus

    print('Batch size', args.batch)

    ####################################################
    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    if args.precache is None:
        # Load individual files (all objects)
        ds_train, ds_validation, ds_testing, n_classes = load_data_set_by_folds(args, objects = list(range(10)))
    else:
        # Load pre-cached data
        ds_train, ds_validation, ds_testing, n_classes = load_precached_folds(args)

    ####################################################
    # Build the model
    if args.verbose >= 3:
        print('Building network')

    image_size=args.image_size[0:2]
    nchannels = args.image_size[2]

    # Network config
    # NOTE: this is very specific to our implementation of create_cnn_classifier_network()
    #   List comprehension and zip all in one place (ugly, but effective).
    #   Feel free to organize this differently
    #
    #  Our implementation requries 2-tuples for kernel size, pool size and stride size
    #
    conv_layers = [{'filters': f, 'kernel_size': (s,s), 'pool_size': (p,p), 'strides': (p,p), 'batch_normalization': args.batch_normalization}
                   if p > 1 else {'filters': f, 'kernel_size': (s,s), 'pool_size': None, 'strides': None, 'batch_normalization': args.batch_normalization}
                   for s, f, p, in zip(args.conv_size, args.conv_nfilters, args.pool)]
    
    dense_layers = [{'units': i, 'batch_normalization': args.batch_normalization} for i in args.hidden]
    
    print("Dense layers:", dense_layers)
    print("Conv layers:", conv_layers)

    # Create the network
    if multi_gpus > 1:
        # Multiple GPUs
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            # Build network: you must provide your own implementation
            model = create_cnn_classifier_network(image_size,
                                                  nchannels,
                                                  conv_layers=conv_layers,
                                                  dense_layers=dense_layers,
                                                  p_dropout=args.dropout,
                                                  p_spatial_dropout=args.spatial_dropout,
                                                  lambda_l2=args.L2_regularization,
                                                  lrate=args.lrate, n_classes=n_classes,
                                                  loss='sparse_categorical_crossentropy',
                                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                                  padding=args.padding,
                                                  flatten=False,
                                                  conv_activation=args.activation_conv,
                                                  dense_activation=args.activation_dense)
    else:
        # Single GPU
        # Build network: you must provide your own implementation
        model = create_cnn_classifier_network(image_size,
                                              nchannels,
                                              conv_layers=conv_layers,
                                              dense_layers=dense_layers,
                                              p_dropout=args.dropout,
                                              p_spatial_dropout=args.spatial_dropout,
                                              lambda_l2=args.L2_regularization,
                                              lrate=args.lrate, n_classes=n_classes,
                                              loss='sparse_categorical_crossentropy',
                                              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                                              padding=args.padding,
                                              flatten=False,
                                              args = args)
        
    
    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase

    # Plot the model
    if args.render:
        render_fname = '%s_model_plot.png'%fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #####
    # Start wandb
    run = wandb.init(project=args.project, name='%s_R%d'%(args.label,args.rotation), notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'model architecture': wandb.Image(render_fname)})

            
    #####
    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                      min_delta=args.min_delta, monitor=args.monitor)
    cbs.append(early_stopping_cb)

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used
    history = model.fit(ds_train,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        use_multiprocessing=True, 
                        verbose=args.verbose>=2,
                        validation_data=ds_validation,
                        validation_steps=None,
                        callbacks=cbs)

    # Done training

    # Generate results data
    results = {}

    # Validation set
    print('#################')
    print('Validation')
    results['args'] = args
    results['predict_validation'] = model.predict(ds_validation)
    results['predict_validation_eval'] = model.evaluate(ds_validation)
    wandb.log({'final_val_loss': results['predict_validation_eval'][0]})
    wandb.log({'final_val_sparse_categorical_accuracy': results['predict_validation_eval'][1]})

    # Test set
    if ds_testing is not None:
        print('#################')
        print('Testing')
        results['predict_testing'] = model.predict(ds_testing)
        results['predict_testing_eval'] = model.evaluate(ds_testing)
        wandb.log({'final_test_loss': results['predict_testing_eval'][0]})
        wandb.log({'final_test_sparse_categorical_accuracy': results['predict_testing_eval'][1]})

    # Training set
    print('#################')
    print('Training')
    results['predict_training'] = model.predict(ds_train)
    results['predict_training_eval'] = model.evaluate(ds_train)

    wandb.log({'final_train_loss': results['predict_training_eval'][0]})
    wandb.log({'final_train_sparse_categorical_accuracy': results['predict_training_eval'][1]})

    results['history'] = history.history

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        model.save("%s_model"%(fbase))

    wandb.finish()

    return model


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    #n_physical_devices = 0

    if args.verbose >= 3:
        print('Arguments parsed')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')


    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

        #if args.verbose >= 3:
        #print('GPU configured')

    if args.check:
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Set number of threads, if it is specified
        if args.cpus_per_task is not None:
            tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
            tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

        execute_exp(args, multi_gpus=n_visible_devices)
        
