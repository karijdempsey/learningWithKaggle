import itertools
import os
import glob
import json

import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Image
import skimage.io
import numpy as np
import tifffile as t
import rasterio
import scipy.misc
from plotly import __version__
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) # connect to notebook

from scipy.misc import imread
   
    
def display_rnd_image(src_dir):
    rand_img = np.random.choice(glob.glob(src_dir + '/*.png'))
    print(rand_img)
    img = cv2.imread(rand_img)
    plt.imshow(img)
    plt.show()
    

# === Evaluation ===
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(10,12))
    plt.show()

    
def kari_plot(preds, short_validation_filenames):
    """ Plot prediction probability (for each image and class) - area=probability
        (See experiments_crops_20170815/1.1-mph_as_kd-baseline-simple_CNN_from_scratch.ipynb 
        for example usage/interpretation)"""
    
    oil_and_gas_infrastructure_probs = go.Bar(
        x=range(preds.shape[0]),
        y=preds[:,0],
        name='oil_and_gas_infrastructure',
        text=short_validation_filenames
    )
    other_probs = go.Bar(
        x=range(preds.shape[0]),
        y=preds[:,1],
        name='other',
        text=short_validation_filenames,
    )
    turbine_probs = go.Bar(
        x=range(preds.shape[0]),
        y=preds[:,2],
        name='turbine',
        text=short_validation_filenames,
    )

    data = [oil_and_gas_infrastructure_probs, other_probs, turbine_probs]
    layout = go.Layout(
        barmode='stack'
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')
    
    
def display_image(img_path):
    print(img_path)
    img = cv2.imread(img_path)
    plt.figure(figsize=(10,12))
    plt.imshow(img)
    plt.show()
    
def display_random_good_prediction(preds, preds_filenames, base_feature_dir, class_col_idx, prob_threshold=0.99):
    idxs = np.where(preds[:, class_col_idx] > prob_threshold)
    """ 
    preds = np.array([[oil_and_gas_infrasture_prob, other_prob, turbine_prob],
                      [...]]
    preds_filenames = ['oil_and_gas_infrastructure/S1A_IW_GRDH_1SDV_20161019T060558_
                        20161019T060623_013555_015B30_0BEA_terrain_correction_46.png', ...]
    base_feature_dir = '/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/'
    
                      
    (See example usage in experiments_crops_20170815/1.1-mph_as_kd-baseline-simple_CNN_from_scratch.ipynb) """
    
    random_idx = np.random.choice(idxs[0])  # idxs is a tuple, hence the [0]
    
    class_dir_and_filename = preds_filenames[random_idx]
    
    img_path = base_feature_dir + "/" + class_dir_and_filename
    print("Number of samples found: {0}".format(len(idxs[0])))
    print(preds[random_idx,])
    display_image(img_path)
    
def display_random_bad_prediction(preds, preds_filenames, base_feature_dir, class_col_idx, prob_threshold=0.01):
    idxs = np.where(preds[:, class_col_idx] < prob_threshold)
    """ 
    preds = np.array([[oil_and_gas_infrasture_prob, other_prob, turbine_prob],
                      [...]]
    preds_filenames = ['oil_and_gas_infrastructure/S1A_IW_GRDH_1SDV_20161019T060558_
                        20161019T060623_013555_015B30_0BEA_terrain_correction_46.png', ...]
    base_feature_dir = '/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/'
    
                      
    (See example usage in experiments_crops_20170815/1.1-mph_as_kd-baseline-simple_CNN_from_scratch.ipynb) """
    
    random_idx = np.random.choice(idxs[0])  # idxs is a tuple, hence the [0]
    
    class_dir_and_filename = preds_filenames[random_idx]
    
    img_path = base_feature_dir + "/" + class_dir_and_filename
    print("Number of samples found: {0}".format(len(idxs[0])))
    print(preds[random_idx,])
    display_image(img_path)
    
        
def display_random_uncertain_prediction(preds, preds_filenames, base_feature_dir, class_col_idx, le_threshold=0.55, gt_threshold=0.45):
    idxs = np.where((preds[:, class_col_idx] < le_threshold) & (preds[:, class_col_idx] > gt_threshold))
    """ 
    preds = np.array([[oil_and_gas_infrasture_prob, other_prob, turbine_prob],
                      [...]]
    preds_filenames = ['oil_and_gas_infrastructure/S1A_IW_GRDH_1SDV_20161019T060558_
                        20161019T060623_013555_015B30_0BEA_terrain_correction_46.png', ...]
    base_feature_dir = '/home/ubuntu/data/sar/experiment_crops_20170815/train/50x50/'
    
                      
    (See example usage in experiments_crops_20170815/1.1-mph_as_kd-baseline-simple_CNN_from_scratch.ipynb) """
    
    random_idx = np.random.choice(idxs[0])  # idxs is a tuple, hence the [0]
    
    class_dir_and_filename = preds_filenames[random_idx]
    
    img_path = base_feature_dir + "/" + class_dir_and_filename
    print("Number of samples found: {0}".format(len(idxs[0])))
    print(preds[random_idx,])
    display_image(img_path)
# ==================
