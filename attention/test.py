import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns

def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.title(file_name)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()

def draw_cm(cm, class_names):
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(cm)#, annot=True, ax=ax, fmt='.2f');  # annot=True to annotate cells
    # labels, title and ticks
    '''
    ax.set_xlabel('Predicted', fontsize=16)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=12)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=16)
    ax.yaxis.set_ticklabels(class_names, fontsize=12)
    plt.yticks(rotation=0)
    '''

    imgSave("./", "test")

Channel_location = ["A_Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
                        "T7", "TP9", "CP5", "CP1", "PZ", "P3", "P7", "O1",
                        "OZ", "O2", "P4", "P8", "TP10", "CP6", "CP2", "CZ",
                        "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2",
                        "B_Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
                        "T7", "TP9", "CP5", "CP1", "PZ", "P3", "P7", "O1",
                        "OZ", "O2", "P4", "P8", "TP10", "CP6", "CP2", "CZ",
                        "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"]
inputs = torch.rand(64,64)
inputs = inputs.numpy()
#print(Channel_location[np.array([1,23])])
#print(Channel_location[np.array([3, 4])])
draw_cm(inputs, class_names=Channel_location[0:22])