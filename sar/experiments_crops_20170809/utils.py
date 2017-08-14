import itertools
import os
import glob

import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Image
import skimage.io
import numpy as np
import tifffile as t
import rasterio
import scipy.misc


# === CONVERT TIFF TO PNG (follows: https://github.com/karijdempsey/learningWithKaggle/blob/master/sar/MPH/1.0-mph-convert_display_images.ipynb) ===

    
def tif2png(src_dir, dest_dir):
    print src_dir, len([item for item in os.listdir(src_dir)])
    count = 0
    for filename in os.listdir (src_dir):
        data_path = os.path.join(src_dir, filename)    
        with rasterio.open(data_path, mode="r") as raster:
            img_array = raster.read(1)
        count +=1
#     plt.imshow(img_array)
#     plt.show()
        out_path = dest_dir + filename
        out_path = out_path.replace('.tif', '.png')
        scipy.misc.imsave(out_path, img_array)
    
    print dest_dir, count, len([item for item in os.listdir(dest_dir)])
    
    
def display_rnd_image(src_dir):
    rand_img = np.random.choice(glob.glob(src_dir + '/*.png'))
    print rand_img
    img = cv2.imread(rand_img)
    plt.imshow(img)
    plt.show()
    
def display_image(img_path):
    print img_path
    img = cv2.imread(img_path)
    plt.figure(figsize=(10,12))
    plt.imshow(img)
    plt.show()
    
# Usage Examples:
#convert_files(root_folder + "/train/oil/crops_oil_50x50", root_folder + "/exp/train/oil/")
#convert_files(root_folder + "/train/other/crops_other_50x50", root_folder + "/exp/train/other/")
#convert_files(root_folder + "/train/turbine/crops_turbine_50x50", root_folder + "/exp/train/turbine/")
#convert_files(root_folder + "/validate/oil/crops_oil_50x50", root_folder + "/exp/valid/oil/")
#convert_files(root_folder + "/validate/other/crops_other_50x50", root_folder + "/exp/valid/other/")
#convert_files(root_folder + "/validate/turbine/crops_turbine_50x50", root_folder + "/exp/valid/turbine/")

#display_rnd_image(root_folder + "/exp/train/oil/")
#display_rnd_image(root_folder + "/exp/train/other/")
#display_rnd_image(root_folder + "/exp/train/turbine/")
#display_rnd_image(root_folder + "/exp/valid/oil/")
#display_rnd_image(root_folder + "/exp/valid/other/")
#display_rnd_image(root_folder + "/exp/valid/turbine/")

# =============================================


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

# ==================