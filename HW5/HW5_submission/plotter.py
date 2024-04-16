import os
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
import sys
import time
import fnmatch
import scipy 
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from hw5_base import *

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
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)
    return results


def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Plotter Function', fromfile_prefix_chars='@')
    
    # Path instructions handler 
    parser.add_argument('--path', type = str, default = '/home/cs504305/deep_learning_practice/homework/hw5/results',help = 'Provide path to the result file')
    # parser.add_argument('--base', type = str, default = 'bmi_*results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    parser.add_argument('--cnn_path', type = str, default = '/home/cs504322/homeworks/HW5/results/cnn/run1', help= 'Provide Path to CNN Network')
    #parser.add_argument('--srnn_path', type = str, default = '/home/cs504322/homeworks/HW5/results/rnn_pool/run1', help= 'Provide path to SRNN Network')
    parser.add_argument('--rnn_pool_path', type = str, default = '/home/cs504322/homeworks/HW5/results/rnn_pool/run4', help= 'Provide path to RNN pool Network')
    parser.add_argument('--base', type = str, default = 'image*rot*_results.pkl', help= 'Provide Filename structure for results, may use * for wildcard')
    # Print Figures
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
    # types of outputs  
    #parser.add_argument('--dropout', action='store_true', help='Plot results for dropout set')
    #parser.add_argument('--regularization', action='store_true', help='Plot results for L1 regularization set')

    # 
    return parser
    
    
def plot_results(epochs = None, data1 = None, data2 = None, data3 = None, data4 = None, data5 = None, xlabel = None, ylabel= None, title = None):
    """
    This function builds plots based on the inputs given

    """
    if data1 is not None:
        (data, label1) = data1
        #print(data)
        plt.plot(range(0,len(data)), data, label = label1 )

    if data2 is not None:
        (data, label2) = data2
        plt.plot(range(0,len(data)), data, label = label2)

    if data3 is not None:
        (data, label3) = data3
        plt.plot(range(0,len(data)), data, label = label3)

    if data4 is not None:
        (data, label4) = data4
        plt.plot(range(0,len(data)), data, label = label4)
    
    if data5 is not None:
        (data, label5) = data5
        plt.plot(range(0,len(data)), data, label = label5)

    plt.legend()
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        #print(title)
        plt.title(title)

    plt.savefig("plots/Fig1_%s.png"%title)
    print('Figure Saved: %s'%title)
    plt.clf()
    
    return 0

def plot_hist(
    data1 = None, 
    data2 = None, 
    title = None
):
    

    # Create bar positions for UNET and AUTO
    bar_width = 0.35
    index = np.arange(len(data1))

    # Creating the bar chart
    plt.bar(index, data1, bar_width, label='cnn', color='b')
    plt.bar(index + bar_width, data2, bar_width, label='rnn', color='r')

    # Adding labels and title
    plt.xlabel('Rotations')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Bar Chart')
    plt.ylim(0.1, 1.5)
    plt.xticks(index + bar_width / 2, ('1', '2', '3', '4', '5'))

    # Adding a legend
    plt.legend()

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
    for i, result in enumerate(results):
        train_accuracy.append(result['history']['sparse_categorical_accuracy'])
        val_accuracy.append(result['history']['val_sparse_categorical_accuracy'])
        test_accuracy.append(result['predict_testing_eval'][1])
    
    return train_accuracy, val_accuracy, test_accuracy

def plot_results_new(data1, label1 = 'Hello' , data2 = None, label2 = None , data3 = None, label3 = None, data4 = None, label4 = None, graph_params = None):
    
    '''
    We are taking the data as a set of all the results for each model including all rotations and plotting them
    
    ''' 
    
    for data in data1:
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

def plot_scatter(data1, data2, graph_params):
    
    # Plot the scatter plot of the data
    plt.scatter(data1, data2, c=['red', 'blue', 'green', 'yellow', 'orange'], alpha = 0.5, s = [100, 100, 100, 100, 100])
    
    for i in range(len(data1)):
        plt.text(data1[i], data2[i], str(i), fontsize=12)  
    
    if graph_params is not None:
        plt.title(graph_params['title'])
        plt.xlabel(graph_params['xlabel'])
        plt.ylabel(graph_params['ylabel'])

    plt.savefig("plots/Fig3_%s.png"%graph_params['title'])
    print('Figure Saved: %s'%graph_params['title'])
    plt.clf()
    
    return 0

if __name__ == "__main__":
    
    # Hide GPU from visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    parser = create_parser()
    args = parser.parse_args()

    '''
    Handle the arguments to provide inputs to the function to build figures for HW 3
    
    1. We have path variables to provide the path to the results files
    2. We have deep_base and deep_path to provide the path to the deep learning results files
    3. We need to get the results and plot them 
    ''' 
    

    # Get the results for the all models
    cnn = read_all_rotations(args.cnn_path, args.base)
    #srnn = read_all_rotations(args.srnn_path, args.base)
    rnn_pool = read_all_rotations(args.rnn_pool_path, args.base)
    
    # Make the results in plottable format
    
    cnn_train_accuracy, cnn_val_accuracy, cnn_test_accuracy = prepare_result(cnn)
    #srnn_train_accuracy, srnn_val_accuracy, srnn_test_accuracy = prepare_result(srnn)
    rnn_pool_train_accuracy, rnn_pool_val_accuracy, rnn_pool_test_accuracy = prepare_result(rnn_pool)
    
    print(rnn_pool_test_accuracy)
    # make_test_predictions(deep,shallow) 

    '''
  Plot Figures 1 2 3 
  
  We are plotting a total of 3 figures here 
  
    1. Training set accuracy as a function of epoch for each rotation.
    2. Validation set accuracy as a function of epoch for each rotation.
    3. Scatter plot of the test set accuracy for each rotation.
  
  We had to make multiple iterations of plot_results because we have some amount of manual inputs going into the function here. 
  Figures and Plots need to be personalized for this task
    '''
    if args.plot:
    
        # Plot Figure 1
        graph_params = {'title': 'Training Set Accuracy', 'xlabel': 'Epochs', 'ylabel': 'Accuracy'}
        plot_results_new(cnn_train_accuracy, 'CNN', rnn_pool_train_accuracy, 'RNN_Pool', graph_params = graph_params)
        
        # Plot Figure 2
        graph_params = {'title': 'Validation Set Accuracy', 'xlabel': 'Epochs', 'ylabel': 'Accuracy'}
        plot_results_new(cnn_val_accuracy, 'CNN', rnn_pool_val_accuracy, 'RNN_Pool', graph_params = graph_params)
        
        # Figure 3
        
        #Compute Test Accuracy(CNN) - Test Accuracy(SRNN)
        '''
        for i in range(len(cnn_test_accuracy)):
            cnn_test_accuracy[i] = cnn_test_accuracy[i] - srnn_test_accuracy[i]
            rnn_pool_test_accuracy[i] = rnn_pool_test_accuracy[i] - srnn_test_accuracy[i]
        '''
        plot_hist(cnn_test_accuracy, rnn_pool_test_accuracy, title = "Bar plot of test set accuracy for both models.")

        graph_params = {'title': 'Scatter plot of test set accuracy: RNN vs CNN', 'xlabel': 'CNN', 'ylabel': 'RNN'}
        plot_scatter(cnn_test_accuracy, rnn_pool_test_accuracy, graph_params = graph_params)
        print("cnn_test_accuracy:",cnn_test_accuracy)
        print("rnn_pool_test_accuracy:",rnn_pool_test_accuracy)