'''
Plotter helps us plot beautiful figures from the results file

'''

import os
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import argparse
import sys
import time
import fnmatch
import wandb
from hw1_base_skel import check_args

#################################################################
# Default plotting parameters
FONTSIZE = 14
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################

wandb.init(project="HW1", entity="vineelpalla14")

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
    parser.add_argument('--path', type = str, default = '/home/cs504322/homeworks/HW1/results/r1/',help = 'Provide path to the result file')
    parser.add_argument('--base', type = str, default = 'bmi__ddtheta_0_hidden_500_500_JI_Ntraining_2_rotation_10_results.pkl', help= 'Provide Filename structure, may use * for wildcard')
    
    # Print Figure 1 only
    parser.add_argument('--single_file', action='store_true', help='Perform on a single file')
    
    
    # 
    return parser
    
def visualize_predictions(fbase):
    """
    Visualizes the comparison between actual and predicted values from a result file.

    :param file_path: Path to the directory containing the result file.
    :param file_name: Name of the result file to visualize.
    """
    # Construct the full file path
    full_path = os.path.join(args.path, args.base)
    
    # Load the result data
    with open(full_path, "rb") as file:
        results = pickle.load(file)

    # Plotting
    actual = results['outs_testing'][:1000]
    predicted = results['predict_testing'][:1000]
    time_steps = results['time_testing'][:1000]

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, actual, label='Actual Acceleration', linestyle='-')
    plt.plot(time_steps, predicted, label='Predicted Acceleration', linestyle='--')
    plt.title('Prediction Accuracy')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.savefig("Prediction_Comparison.png")
    wandb.log({"Prediction_Comparison": wandb.Image(plt)})
    plt.close()

def display_train_folds(dir, base):
    """
    Plots the training, validation, and test FVAF values across different folds from result files.

    :dir: Directory containing result files.
    :base: To match result files.
    """
    # Temporary list to perform sorting
    temp_train = []
    temp_val = []
    temp_test = []
    
    # Lists for plotting the data 
    train_fvaf = []
    val_fvaf = []
    test_fvaf = []
    folds = []
    # initialize 
    old_fold = 0
    
    # Read all the results
    results = read_all_rotations(dir, base)

    
    # Iterate through the results
    for result in results:
        history = result['history']
        new_fold = (len(result['folds']['folds_training']))
        
        if(new_fold!= old_fold):
            folds.append(new_fold)
            if (old_fold != 0):
                #print(new_fold)
                train_fvaf.append(np.mean(temp_train))
                val_fvaf.append(np.mean(temp_val))
                test_fvaf.append(np.mean(temp_test))
                
                
                temp_train = []
                temp_test = []
                temp_val = []
                
            old_fold = new_fold
        
        temp_train.append(result['predict_training_eval'][1])
        temp_val.append(result['predict_validation_eval'][1])
        temp_test.append(result['predict_testing_eval'][1])
    
    train_fvaf.append(np.mean(temp_train))
    val_fvaf.append(np.mean(temp_val))
    test_fvaf.append(np.mean(temp_test))                

    # Sort by fold index for coherent plotting

    sorted_indices = np.argsort(folds)
    folds = np.array(folds)[sorted_indices]
    train_fvaf = np.array(train_fvaf)[sorted_indices]
    val_fvaf = np.array(val_fvaf)[sorted_indices]
    test_fvaf = np.array(test_fvaf)[sorted_indices]

    print(folds)

    # plot the figure 
    
    plt.plot(folds, train_fvaf, label = 'Training FVAF')
    plt.plot(folds, val_fvaf, label = 'Validation FVAF')
    plt.plot(folds, test_fvaf, label = 'Testing FVAF')
    plt.legend()
    plt.ylabel('Avg. FVAF')
    plt.xlabel('Folds')
    plt.savefig("Fig2.png")

    wandb.log({"Fig2": wandb.Image(plt)})
    plt.close()
    
    
if __name__ == "__main__":
    
    parser = create_parser()
    args = parser.parse_args()
    
    visualize_predictions(args)

    args.base = "bmi__ddtheta_0_hidden*_results.pkl"
    display_train_folds(args.path, args.base)

