import os
import fnmatch
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt 
import scipy 


# Initialize the wandb project
wandb.init(project="HW2", entity="vineelpalla14")


def load_data(directory_name, filename_pattern):
    '''Load and return data from files in the given directory that match the filename pattern.'''
    
    # List all files in the directory that match the filename pattern
    matching_files = fnmatch.filter(os.listdir(directory_name), filename_pattern)
    matching_files.sort()
    loaded_data = []

    # Open each file, load its contents with pickle, and add the result to the list
    for filename in matching_files:
        with open(os.path.join(directory_name, filename), "rb") as file_pointer:
            data = pickle.load(file_pointer)
            loaded_data.append(data)
    
    return loaded_data




def load_results(directory, file_pattern):
    '''
    Load results from specified directory and aggregate FVAF (Feature Vector Angle Frequency) metrics 
    across training, validation, and testing datasets for each fold.
    
    :param directory: Directory containing result files
    :param file_pattern: Pattern to match result files
    '''
    train_evaluations = []
    validation_evaluations = []
    test_evaluations = []

    # Aggregation lists for FVAF metrics
    train_metrics_aggregate = []
    validation_metrics_aggregate = []
    test_metrics_aggregate = []
    fold_counts = []
    
    # Variables for tracking fold changes
    current_fold_count = 0
    
    # Load results matching the given pattern
    results = load_data(directory, file_pattern)

    # Process each result
    for result in results:
        history = result['history']
        fold_count = len(result['folds']['folds_training'])

        # Detect a change in the fold count, indicating a new set of results
        if fold_count != current_fold_count:
            if current_fold_count != 0:
                # Aggregate the metrics for the previous fold
                train_metrics_aggregate.append(np.mean(train_evaluations))
                validation_metrics_aggregate.append(np.mean(validation_evaluations))
                test_metrics_aggregate.append(np.mean(test_evaluations))
                
                # Reset temporary lists for the new fold
                train_evaluations = []
                validation_evaluations = []
                test_evaluations = []
            
            fold_counts.append(fold_count)
            current_fold_count = fold_count
        
        # Accumulate evaluations for the current fold
        train_evaluations.append(result['predict_training_eval'][1])
        validation_evaluations.append(result['predict_validation_eval'][1])
        test_evaluations.append(result['predict_testing_eval'][1])
    
    # Aggregate the final set of evaluations
    train_metrics_aggregate.append(np.mean(train_evaluations))
    validation_metrics_aggregate.append(np.mean(validation_evaluations))
    test_metrics_aggregate.append(np.mean(test_evaluations))

    # Organize data by fold count
    sorted_fold_indices = np.argsort(fold_counts)
    fold_counts_sorted = np.array(fold_counts)[sorted_fold_indices]
    train_fvaf_sorted = np.array(train_metrics_aggregate)[sorted_fold_indices]
    validation_fvaf_sorted = np.array(validation_metrics_aggregate)[sorted_fold_indices]
    test_fvaf_sorted = np.array(test_metrics_aggregate)[sorted_fold_indices]

    return train_fvaf_sorted, validation_fvaf_sorted, test_fvaf_sorted, fold_counts_sorted

def plot_data(folds, *data_series, xlabel=None, ylabel=None, title=None):
    """
    Plots multiple data series on the same plot.

    Parameters:
    - folds: The x-axis values for all data series.
    - *data_series: Variable length list of tuples, where each tuple is (data, label).
    - xlabel, ylabel, title: Plot labeling options.
    """
    for data, label in data_series:
        plt.plot(folds, data, label=label)
    
    plt.legend()
    
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if title:
        plt.title(title)
    
    plt.savefig(f"plots/{title}.png")
    wandb.log({f"Fig:{title}.png": wandb.Image(plt)})
    plt.clf()

if __name__ == "__main__":
    

    directory = '/home/cs504322/homeworks/HW2/results/part_1/'
    filenames = 'bmi_*results.pkl'

    dropout_directory = '/home/cs504322/homeworks/HW2/results/part_2/'
    dropout_files = ["bmi_*_dropout_0.1_results.pkl", "bmi_*_dropout_0.25_results.pkl", "bmi_*_dropout_0.45_results.pkl", "bmi_*_dropout_0.65_results.pkl", "bmi_*_dropout_0.85_results.pkl"]

    regularization_directory = '/home/cs504322/homeworks/HW2/results/part_3_1/'
    regularization_files = ["bmi_*_regularization_10_results.pkl", "bmi_*_regularization_0.001_results.pkl", "bmi_*_regularization_0.0001_results.pkl", "bmi_*_regularization_1e-05_results.pkl", "bmi_*_regularization_1e-06_results.pkl"]
    
    fvaf_results = []
    dropout_fvaf_results = []
    reg_fvaf_results = []
    
    

    #Fig 1
    # get results for all the normal runs
    training_fvaf, validation_fvaf, testing_fvaf, fold_counts = load_results(directory, filenames)
    # Organize FVAF metrics into a dictionary for easy access
    fvaf_metrics = {
        'train': training_fvaf,
        'val': validation_fvaf,
        'test': testing_fvaf
    }
    # Prepare a list to accumulate FVAF metric dictionaries
    
    fvaf_results.append(fvaf_metrics)
    
    # Plot curves for without regularization
    data_series = [(fvaf_results[0]['train'], 'Training'), (fvaf_results[0]['val'], 'Validation')]
    plot_data(
        fold_counts, 
        *data_series,
        xlabel='Folds', 
        ylabel='Mean FVAF', 
        title='Performance without Regularization'
    )

    #Fig 2
    
    for i,fp in enumerate(dropout_files):
        training_fvaf, validation_fvaf, testing_fvaf, fold_counts = load_results(dropout_directory, fp)
        # Organize FVAF metrics into a dictionary for easy access
        fvaf_metrics = {
            'train': training_fvaf,
            'val': validation_fvaf,
            'test': testing_fvaf
        }
        dropout_fvaf_results.append(fvaf_metrics)
        #max_d.append(np.max(fvaf_metrics['val']))
        
        
    # plot results for dropout 
    # Example for dropout results plotting with the new plot_data function
    dropout_labels = [0.1, 0.25, 0.45, 0.65, 0.85]
    data_series = [(dropout_fvaf_results[i]['val'], f'Dropout {dropout_labels[i]}') for i in range(len(dropout_fvaf_results))]
    plot_data(fold_counts, *data_series, xlabel='Folds', ylabel='Mean FVAF', title='Dropout')


    #Fig 3
    
    for i,fp in enumerate(regularization_files):
        training_fvaf, validation_fvaf, testing_fvaf, fold_counts = load_results(regularization_directory, fp)
        # Organize FVAF metrics into a dictionary for easy access
        fvaf_metrics = {
            'train': training_fvaf,
            'val': validation_fvaf,
            'test': testing_fvaf
        }
        #max_l.append(np.max(fvaf_metrics['val']))
        reg_fvaf_results.append(fvaf_metrics)
        
        
        # plot results for all regularization runs 
    L1 = [10, 0.001, 0.0001, 0.00001, 0.000001]
    data_series = [(reg_fvaf_results[i]['val'], f'L1 {L1[i]}') for i in range(len(reg_fvaf_results))]
    plot_data(fold_counts, *data_series, xlabel='Folds', ylabel='Mean FVAF', title='Regularization')

    
    # Perform t-test 
 
    res = []
    for fp1 in dropout_fvaf_results:
        res.append(fp1['val'])
        res1 = np.array(res)
        best_dropout = np.argmax(res1, axis = 0)
    print("best dropout is :",best_dropout)
        
    res = []
    for fp2 in reg_fvaf_results:
        res.append(fp2['val'])
        res1 = np.array(res)
        best_reg = np.argmax(res1, axis = 0)
    print("best regularization is :",best_reg)
       
    #Fig 4

    test_scores_dropout = []
    test_scores_regularization = []
    for i, n in enumerate(best_dropout):
        test_scores_dropout.append(dropout_fvaf_results[n]['test'][i])
    
    for i, n in enumerate(best_reg):
        test_scores_regularization.append(reg_fvaf_results[n]['test'][i])
    # Retrieve normal scenario test scores for comparison
    test_scores_normal = fvaf_results[0]['test']

    
    data_series = [(test_scores_dropout, "Dropout"), (test_scores_normal, "No regularization"), (test_scores_regularization, "L1 Regularization")]
    plot_data(fold_counts, *data_series, xlabel = "Folds", ylabel = "Best fvaf", title = "best hyper-parameter set" )
    
    print("T-Tests:") 
    print("For training set size 1")
    print("-----------------------------------------------")
    print("-----------------------------------------------")

    # Loading predicted testing evaluations
    without_L1_fvaf = []
    without_L1 = load_data(directory, "bmi*Ntraining_1_*results.pkl")
    for fp3 in without_L1:
        without_L1_fvaf.append(fp3['predict_testing_eval'][1])


    with_dropout_fvaf = []
    with_dropout = load_data(dropout_directory, "bmi*Ntraining_1_*dropout_0.25*results.pkl")
    for fp4 in with_dropout:
        with_dropout_fvaf.append(fp4['predict_testing_eval'][1])

    with_L1_fvaf = []
    with_L1 = load_data(regularization_directory, "bmi*Ntraining_1_*regularization_1e-06_results.pkl")
    for fp5 in with_L1:
        with_L1_fvaf.append(fp5['predict_testing_eval'][1])
    '''
    print("results")
    print(without_L1_fvaf)
    print(with_dropout_fvaf)
    print(with_L1_fvaf)
    ''' 
    print("T-test between without-regularization and with-dropout:")
    T_test = scipy.stats.ttest_rel(without_L1_fvaf, with_dropout_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(without_L1_fvaf)-np.mean(with_dropout_fvaf))

    print("T-test between without-regularization and L1-regularization:")
    T_test = scipy.stats.ttest_rel(without_L1_fvaf, with_L1_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(without_L1_fvaf)-np.mean(with_L1_fvaf))

    print("T-test between with-dropout and L1-regularization:")
    T_test = scipy.stats.ttest_rel(with_dropout_fvaf, with_L1_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(with_dropout_fvaf)-np.mean(with_L1_fvaf))


    print("-----------------------------------------------")
    print("-----------------------------------------------")
    
    
    print("T-Tests:") 
    print("For training set size 18")
    print("-----------------------------------------------")
    print("-----------------------------------------------")

    # Loading predicted testing evaluations
    without_L1_fvaf = []
    without_L1 = load_data(directory, "bmi*Ntraining_18_*results.pkl")
    for fp3 in without_L1:
        without_L1_fvaf.append(fp3['predict_testing_eval'][1])


    with_dropout_fvaf = []
    with_dropout = load_data(dropout_directory, "bmi*Ntraining_18_*dropout_0.25*results.pkl")
    for fp4 in with_dropout:
        with_dropout_fvaf.append(fp4['predict_testing_eval'][1])

    with_L1_fvaf = []
    with_L1 = load_data(regularization_directory, "bmi*Ntraining_18_*regularization_1e-06_results.pkl")
    for fp5 in with_L1:
        with_L1_fvaf.append(fp5['predict_testing_eval'][1])
        
    print("T-test between without-regularization and with-dropout:")
    T_test = scipy.stats.ttest_rel(without_L1_fvaf, with_dropout_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(without_L1_fvaf)-np.mean(with_dropout_fvaf))

    print("T-test between without-regularization and L1-regularization:")
    T_test = scipy.stats.ttest_rel(without_L1_fvaf, with_L1_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(without_L1_fvaf)-np.mean(with_L1_fvaf))

    print("T-test between with-dropout and L1-regularization:")
    T_test = scipy.stats.ttest_rel(with_dropout_fvaf, with_L1_fvaf)
    print(T_test)
    print("Differences in means is: ",np.mean(with_dropout_fvaf)-np.mean(with_L1_fvaf))


    print("-----------------------------------------------")
    print("-----------------------------------------------")



    