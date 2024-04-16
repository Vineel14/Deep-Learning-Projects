from core50 import *
from hw3_base import *
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_image_with_probabilities(image, shallow_probs, deep_probs, class_names, index):
    shallow_probs = np.array(shallow_probs)
    deep_probs = np.array(deep_probs)

    plt.figure(figsize=(10, 5))

    # Plot the image with shallow model probabilities
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Shallow Model Probabilities')
    plt.axis('off')
    for i, prob in enumerate(shallow_probs):
        plt.text(5, image.shape[0] - 7 - i * 8, f'{class_names[i]}: {prob:.2f}', color='white', fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')


    # Plot the image with deep model probabilities
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.title('Deep Model Probabilities')
    plt.axis('off')
    for i, prob in enumerate(deep_probs):
        plt.text(5, image.shape[0] - 7 - i * 8, f'{class_names[i]}: {prob:.2f}', color='white', fontsize=14,fontweight='bold', verticalalignment='top', horizontalalignment='left')


    plt.tight_layout()
    plt.savefig(f"plots/F4_{index}.png")
    #plt.show()


def getBestModel(args):
    
    #path = args.results_path +'/'+ shallow
    shallow_model = tf.keras.models.load_model("/home/cs504322/homeworks/HW3/results/shallow/run2/image_Csize_3_Cfilters_10_Pool_2_Pad_valid_hidden_50_20_drop_0.250_sdrop_0.200_L1_0.000100_L2_0.001000_LR_0.001000_ntrain_03_rot_02_model")
    deep_model = tf.keras.models.load_model("/home/cs504322/homeworks/HW3/results/deep/run2/image_Csize_3_3_3_3_3_3_3_Cfilters_8_16_32_64_112_128_256_Pool_2_2_2_2_2_2_2_Pad_same_hidden_320_160_16_drop_0.500_sdrop_0.250_L1_0.000100_L2_0.001000_LR_0.001000_ntrain_03_rot_03_model")

    return shallow_model, deep_model

if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    class_names=['plug', 'mobile', 'glass', 'remote']
    shallow_model, deep_model = getBestModel(args)
    print("Models loaded")

    a , b ,testing_data, object_length = load_data_set_by_folds(args, objects=[4,5,6,8], seed=42)
    print("Data loaded")
    shallow_probs = []
    deep_probs = []
    
    for ins, outs in testing_data.take(1):
        print("Shallow model")
        #print(shallow_model.evaluate(ins, outs))
        shallow_probs.append(shallow_model.predict(ins))

        print("Deep model")
        #print(deep_model.evaluate(ins, outs)) 
        deep_probs.append(deep_model.predict(ins))



    print(shallow_probs[0][1])
    print(deep_probs[0][1])
    


    for i in range(10):
        plot_image_with_probabilities(ins[i], shallow_probs[0][i], deep_probs[0][i], class_names, i)
        #print("shallow probs shape 2 ",shallow_probs[].shape)
    


    ########## Figure 5 ################
     

    shallow_directory = '/home/cs504322/homeworks/HW3/results/shallow/run2/'

    deep_directory = '/home/cs504322/homeworks/HW3/results/deep/run2/'
   
    classes = np.array([0,1,2,3])
    
    shallow_probs=[]
    deep_probs=[]
    print(shallow_probs)
    print(deep_probs)


    
    for i in range(5):
        
        deep_model = tf.keras.models.load_model(deep_directory+f"image_Csize_3_3_3_3_3_3_3_Cfilters_8_16_32_64_112_128_256_Pool_2_2_2_2_2_2_2_Pad_same_hidden_320_160_16_drop_0.500_sdrop_0.250_L1_0.000100_L2_0.001000_LR_0.001000_ntrain_03_rot_0{i}_model")
        print("deep model ", i, "Loaded")
        
        k=0

        for ins, outs in testing_data.take(1):
             
            deep_probs.append(deep_model.predict(ins))
            

        deep_probs=np.array(deep_probs)
 
        deep_probs= deep_probs.reshape(10, 4)

        deep_probs = np.argmax(deep_probs, axis=1)
    
        deep_probs= deep_probs.reshape(10,)
        
        
        outs = outs.numpy()

        cm = confusion_matrix(outs, deep_probs, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        disp.plot()
        plt.savefig(f"plots/F5_deep_{i}.png")
        print("figure saved!")

        deep_probs=[]
        #print(outs)

    
    for i in range(5):
        
        shallow_model = tf.keras.models.load_model(shallow_directory+f"image_Csize_3_Cfilters_10_Pool_2_Pad_valid_hidden_50_20_drop_0.250_sdrop_0.200_L1_0.000100_L2_0.001000_LR_0.001000_ntrain_03_rot_0{i}_model")
        print("shallow model ", i, "Loaded")
        
        k=0

        for ins, outs in testing_data.take(1):
             
            shallow_probs.append(deep_model.predict(ins))
            

        shallow_probs=np.array(shallow_probs)
 
        shallow_probs= shallow_probs.reshape(10, 4)

        shallow_probs = np.argmax(shallow_probs, axis=1)
    
        shallow_probs= shallow_probs.reshape(10,)
        
        
        outs = outs.numpy()

        cm = confusion_matrix(outs, shallow_probs, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        disp.plot()
        plt.savefig(f"plots/F5_shallow_{i}.png")
        print("figure saved!")

        shallow_probs=[]
        #print(outs)