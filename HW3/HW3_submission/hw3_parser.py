'''
Advanced Machine Learning, 2024

Argument parser needed by multiple programs.

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import argparse

def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='CORE50-test', help='WandB project name')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    
    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files");
    parser.add_argument('--dataset', type=str, default='/home/fagg/datasets/core50', help='Data set directory')
    parser.add_argument('--problem', type=str, default='condition', help='Problem type [condition, example]')
    parser.add_argument('--image_size', nargs=3, type=int, default=[128,128,3], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--meta_dataset', type=str, default='core50_df.pkl', help='Name of file containing the core 50 metadata')
    parser.add_argument('--precache', type=str, default=None, help='Precached dataset location')

    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")

    # Convolutional parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('--padding', type=str, default='valid', help='Padding type for convolutional layers')
    parser.add_argument('--activation_conv', type=str, default='elu', help='Activation function for convolutional layers')

    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--activation_dense', type=str, default='elu', help='Activation function for dense layers')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--spatial_dropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")

    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")
    
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")
    parser.add_argument('--no_use_py_func', action='store_true', help="False = use py_function in creating the dataset")

    # Image Augmentation: REVISIT TODO
    #parser.add_argument('--rotation_range', type=int, default=0, help="Image Generator: rotation range")
    #parser.add_argument('--width_shift_range', type=int, default=0, help="Image Generator: width shift range")
    #parser.add_argument('--height_shift_range', type=int, default=0, help="Image Generator: height shift range")
    #parser.add_argument('--shear_range', type=float, default=0.0, help="Image Generator: shift range")
    #parser.add_argument('--zoom_range', type=float, default=0.0, help="Image Generator: zoom range")
    #parser.add_argument('--horizontal_flip', action='store_true', help='Image Generator: horizontal flip')
    #parser.add_argument('--vertical_flip', action='store_true', help='Image Generator: vertical flip')

    # Post
    parser.add_argument('--render', action='store_true', default=False , help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    
    return parser

