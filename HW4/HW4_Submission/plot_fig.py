
from chesapeake_loader import *
import os
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
import sys
import time
import fnmatch
import scipy 
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

  
from hw4_base import *

#################################################################
# Default plotting parameters
FONTSIZE = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################

def read_all_rotations(dirname, filebase):
    '''Read results from dirname from files matching filebase'''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    test_predictions = []
    train, testing, validation = [], [], []
    train_accuracy, test_accuracy, val_accuracy = [], [], []
    # Loop over matching files
    for f in sorted(files):
        print(f)
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        train_accuracy.append(r['history']['sparse_categorical_accuracy'])
        val_accuracy.append(r['history']['val_sparse_categorical_accuracy'])
        test_accuracy.append(r['predict_testing_eval'][1])
        
        # test_predictions.append(r['predict_testing'])

        # results.append(r)
    return train_accuracy, test_accuracy, val_accuracy


def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Plotter Function', fromfile_prefix_chars='@')
    
    # Path instructions handler 
    parser.add_argument('--base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for results, may use * for wildcard')
    parser.add_argument('--unet_path', type = str, default = '/home/cs504322/homeworks/HW4/results/deep', help= 'Provide path to UNET Network')
    parser.add_argument('--auto_path', type = str, default = '/home/cs504322/homeworks/HW4/results/shallow', help= 'Provide path to AUTOENCODER Network')
    parser.add_argument('--ds_path', type = str, default = '/home/cs504322/homeworks/HW4/results', help= 'Provide path to Testing Dataset')
    parser.add_argument('--ds_base', type = str, default = '*_test.pkl', help= 'Provide base to Testing Dataset')
    # Print Figures
    parser.add_argument('--plot', action='store_true', help='Plot results')

    # 
    return parser
    
def plot_hist(
    data1 = None, 
    data2 = None, 
    title = "Hello"
):
    if data2 is not None:
        x = np.array([data1,data2])
        x = x.T
    else:
        x = data1
        
    colors = ['red', 'blue']
    plt.hist(x, 7, density=False, histtype='step', color=colors, label = ("shallow", "deep"), fill = True, stacked = False, alpha = 0.5)
    plt.title("Accuracy between Shallow and Deep Networks")
    plt.legend()
    # plt.set_title('stacked bar')
    plt.savefig("plots/Fig3_%s.png"%title)
    plt.clf()
    print('Figure Saved: %s'%title)
    return 0

def prepare_result(results):
    
    '''
    
    This Function takes the result file as an input and prepares the data for plotting
    
    '''
    
    # Create data for plotting 
    
    train_accuracy = []
    val_accuracy = []
    test_accuracy = []
    
    train_accuracy.append(results['history']['sparse_categorical_accuracy'])
    val_accuracy.append(results['history']['val_sparse_categorical_accuracy'])
    test_accuracy.append(results['predict_testing_eval'][1])

    return train_accuracy, val_accuracy, test_accuracy

def plot_results_new(data1, label1 = 'Hello' , data2 = None, label2 = None , data3 = None, label3 = None, data4 = None, label4 = None, graph_params = None):
    
    '''
    We are taking the data as a set of all the results for each model including all rotations and plotting them
    
    ''' 
    
    for data in data1:
        print(data)
        plt.plot(range(0,len(data)), data, label = label1 +'_Rot_' +str(data1.index(data)), alpha = 0.1 + 0.2*data1.index(data), color = 'red')
    if data2 is not None:
        for data in data2:
            plt.plot(range(0,len(data)), data, label = label2 +'_Rot_' +str(data2.index(data)), alpha = 0.1 + 0.2*data2.index(data), color = 'blue')
    if data3 is not None:
        for data in data3:
            plt.plot(range(0,len(data)), data, label = label3 +'_Rot_' +str(data3.index(data)), alpha = 0.1 + 0.2*data3.index(data), color = 'green')
    if data4 is not None:
        for data in data4:
            plt.plot(range(0,len(data)), data, label = label4 +'_Rot_' +str(data4.index(data)), alpha = 0.1 + 0.2*data4.index(data), color = 'yellow')
    
    plt.legend()
    if graph_params is not None:
        plt.title(graph_params['title'])
        plt.xlabel(graph_params['xlabel'])
        plt.ylabel(graph_params['ylabel'])

    plt.savefig("plots/Fig1_%s.png"%graph_params['title'])
    print('Figure Saved: %s'%graph_params['title'])
    plt.clf()
    
    return 0

# Function to plot bar chart
def plot_bar_chart(data1, data2, graph_params):
    n_groups = len(data1)
    index = np.arange(n_groups)
    bar_width = 0.35

    # Plot the bar chart of the data
    plt.bar(index, data1, bar_width, alpha=0.5, color='blue', label='AUTO')
    plt.bar(index + bar_width, data2, bar_width, alpha=0.5, color='red', label='UNET')

    plt.xlabel(graph_params['xlabel'])
    plt.ylabel(graph_params['ylabel'])
    plt.title(graph_params['title'])
    plt.xticks(index + bar_width / 2, ('Rotation 1', 'Rotation 2', 'Rotation 3', 'Rotation 4', 'Rotation 5'))
    plt.legend()

    # Save the figure
    plt.savefig("plots/Fig3_%s.png" % graph_params['title'])
    print('Figure Saved: %s' % graph_params['title'])

    # Clear the figure after saving to prevent overlap on the next plot
    plt.clf()

    return 0



def create_confusion_matrix(dirname, filebase, test_path, test_base):
    
    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    files.sort()
    test_predictions = []
    train, testing, validation = [], [], []
    # Loop over matching files
    for f in files:
        print(f)
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        arr = np.array(r['predict_testing'])
        print(arr.shape)
        test_predictions.append(arr)
        
        # print(len(r['predict_testing']))

    # print(test_predictions[0][0])
    
    files = fnmatch.filter(os.listdir(test_path), test_base)
    files.sort()
    true_labels = []
    
    for f in files:
        print(f)
        fp = open("%s/%s"%(test_path,f), "rb")
        r = pickle.load(fp)
        fp.close()
        true_labels.append(r)
        # print(len(r['test_labels']))
    
    return 0

def build_confusion_matrix():
    
    # files in the directory for the results of the models 
    for i in range(5):
        # LOAD THE MODEL
        
        # LOAD THE DATA
        ds_test = pd.read_pickle('results/ds_test/{}_ds_test.pkl'.format(i))
        print(ds_test.shape)
        #result = model.predict(ds_test)
        #print(result)
        # LOAD MODELS 
        model = load_model('/home/cs504322/homeworks/HW4/results/models/autoencoder/image_Csize_2_2_2_2_2_Cfilters_30_60_90_120_Pool_3_3_3_3_Pad_same_hidden_100_5_LR_0.001000_ntrain_03_rot_04_model')
        model2 = load_model('/home/cs504322/homeworks/HW4/results/models/unet/image_Csize_2_2_2_2_2_Cfilters_30_60_90_120_Pool_3_3_3_3_Pad_same_hidden_100_5_LR_0.001000_ntrain_03_rot_04_model')
        
        # MAKE PREDICTIONS 
        predictions = model.predict(ds_test)
        predictions2 = model2.predict(ds_test)

        # CONVERT TO LABELS
        preds = np.argmax(predictions, axis = 3)
        preds2 = np.argmax(predictions2, axis = 3)

        # GET OUTPUTS 
        y = np.concatenate([y for x, y in ds_test], axis=0)

        y= np.reshape(y, (250*256*256))
        preds = np.reshape(preds,(250*256*256))
        preds2 = np.reshape(preds2,(250*256*256))

        #CREATE CONFUSUION MATRIX
        conf1 = tf.math.confusion_matrix(
            y,
            preds,
            num_classes=None,
            weights=None,
            dtype=tf.dtypes.int32,
            name=None
        )

        conf2 = tf.math.confusion_matrix(
            y,
            preds2,
            num_classes=None,
            weights=None,
            dtype=tf.dtypes.int32,
            name=None
        )

        plt.figure(figsize = (10,7))
        plt.title('AUTO: Confusion Matrix heatmap for Fold 9')
        sns.heatmap(conf1/np.sum(conf1), annot=True, fmt='.2%', cmap='Blues')
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.savefig('heatmap1.png')
        
        plt.figure(figsize = (10,7))
        plt.title('UNET: Confusion Matrix heatmap for Fold 9')
        sns.heatmap(conf2/np.sum(conf2), annot=True, fmt='.2%', cmap='Blues')
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.savefig('heatmap2.png')

    
    return preds, preds2

if __name__ == "__main__":
    
    
    parser = create_parser()
    args = parser.parse_args()

    '''
    Handle the arguments to provide inputs to the function to build figures for HW 4
    
    1. We have path variables to provide the path to the results files
    2. We have unet_path and auto_path to provide the path to the deep learning results files
    3. We need to get the results and plot them for the following:
        1. Model plots for each model
        2. Validation Accuracy vs Training Epochs
        3. Confusion Matrix for each model
        4. Bar Chart of the test accuracy results
        5. Show interesting images for both models
    ''' 
    
  
    # Get the results for the all models
    auto_train_accuracy, auto_test_accuracy, auto_val_accuracy = read_all_rotations(args.auto_path, args.base)
    unet_train_accuracy, unet_test_accuracy, unet_val_accuracy = read_all_rotations(args.unet_path, args.base)
   

    if args.plot:
    
        
        # Plot Figure 2 - Validation Set Accuracy
        graph_params = {'title': 'Validation Set Accuracy', 'xlabel': 'Training Epochs', 'ylabel': 'Accuracy'}
        plot_results_new(auto_val_accuracy, 'Auto', unet_val_accuracy, 'UNET',  graph_params = graph_params)
        
        # Figure 3 Confusion Matrix
        preds , preds2 = build_confusion_matrix()
        
        #Figure 4 - Bar Chart of Test Accuracy
        graph_params = {'title': 'Test Accuracy Bar Chart', 'xlabel': 'Rotations', 'ylabel': 'Test Accuracy'}
        plot_bar_chart(auto_test_accuracy, unet_test_accuracy, graph_params = graph_params)
        
        
        
        
        