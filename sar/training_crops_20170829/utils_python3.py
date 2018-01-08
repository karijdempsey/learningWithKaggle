import itertools
import os
import glob
import json
import random
from shutil import copy
import pathlib

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


# === Preprocessing ===
def split_all_json_into_train_validate(label):
    train_json_data = {}
    validate_json_data = {}

    data_dir = '/home/ubuntu/data/sar/training_crops_20170829/'
    
    # Make output dir for tifs
    #pathlib.Path(data_dir+'/train/50x50/'+label).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(data_dir+'/train/140x140/'+label).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(data_dir+'/train/240x240/'+label).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(data_dir+'/validate/50x50/'+label).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(data_dir+'/validate/140x140/'+label).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(data_dir+'/validate/240x240/'+label).mkdir(parents=True, exist_ok=True)
    
    os.makedirs(data_dir+'/train/50x50/'+label, exist_ok=True)
    os.makedirs(data_dir+'/train/140x140/'+label, exist_ok=True)
    os.makedirs(data_dir+'/train/240x240/'+label, exist_ok=True)
    os.makedirs(data_dir+'/validate/50x50/'+label, exist_ok=True)
    os.makedirs(data_dir+'/validate/140x140/'+label, exist_ok=True)
    os.makedirs(data_dir+'/validate/240x240/'+label, exist_ok=True)
    
    in_json_path = data_dir+'/distance_to_land/train_'+label+'.json'
    out_train_json_path = data_dir+'train/distance_to_land/all_train_'+label+'.json'
    out_validate_json_path = data_dir+'validate/distance_to_land/all_validate_'+label+'.json'
    
    with open(in_json_path) as json_file:
        json_data = json.load(json_file)


    for filename, distance2land in json_data.items():
        added_to_validate = False

        # Move excatly 500 into validation
        if len(validate_json_data) <= 500:
            # with probability of 30% add crop to validate
            if random.random() <= 0.3:
                validate_json_data[filename] = distance2land
                copy(data_dir+'/50x50/'+label+'/'+filename, data_dir+'/validate/50x50/'+label)
                copy(data_dir+'/140x140/'+label+'/'+filename, data_dir+'/validate/140x140/'+label)
                copy(data_dir+'/240x240/'+label+'/'+filename, data_dir+'/validate/240x240/'+label)
                added_to_validate = True

        if not added_to_validate:
            train_json_data[filename] = distance2land
            copy(data_dir+'/50x50/'+label+'/'+filename, data_dir+'/train/50x50/'+label)
            copy(data_dir+'/140x140/'+label+'/'+filename, data_dir+'/train/140x140/'+label)
            copy(data_dir+'/240x240/'+label+'/'+filename, data_dir+'/train/240x240/'+label)

    print(len(train_json_data))
    print(len(validate_json_data))

    with open(out_train_json_path, 'w') as out_file:
        json.dump(train_json_data, out_file)

    with open(out_validate_json_path, 'w') as out_file:
        json.dump(validate_json_data, out_file)
    

    
def tif2png(src_dir, dest_dir):
    print(src_dir, len([item for item in os.listdir(src_dir)]))
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
    
    print(dest_dir, count, len([item for item in os.listdir(dest_dir)]))
    
    
def display_rnd_image(src_dir):
    rand_img = np.random.choice(glob.glob(src_dir + '/*.png'))
    print(rand_img)
    img = cv2.imread(rand_img)
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

# === Add Features ===

#def add_dist2land(feature_dir, class_name, label_one_hot):
#    """ Example usage: experiments_crops_20170815/1.1-as_kd_mph-new_model_new_data_new_feature.ipynb """
#    json_path = "{0}/distance_to_land/{1}.json".format(feature_dir, class_name)
#   
#    labels = []
#    filenames = []
#    crops = []
#    features = []
#    
#    with open(json_path) as json_data:
#        json_train_data = json.load(json_data)
#        for id_, item in json_train_data.iteritems():
#            
#            filenames.append(id_)
#            features.append(item['distance to land'])
#            labels.append(label_one_hot)
#            
#            img_path = "{0}/50x50/{1}".format(feature_dir, id_)
#            img  = imread(img_path)
#            crops.append(img)
#    return labels, filenames, crops, features

def add_dist2land_training_crops_20170829_multiclass(image_size):
    
    train_dir = '/home/ubuntu/data/sar/training_crops_20170829/train/'+image_size+'/'
    valid_dir = '/home/ubuntu/data/sar/training_crops_20170829/validate/'+image_size+'/'
    train_class = []           
    train_filename = []
    train_crops = []
    train_feature = []

    valid_class = []
    valid_filename = []
    valid_crops = []
    valid_feature = []

    train_class_desc = 'oil_and_gas_infrastructure'
    train_class_array = [1,0,0]
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_oil_and_gas_infrastructure.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0,1,0]
    train_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_turbine.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0,0,1]
    train_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_other.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    valid_class_array = [1,0,0]
    valid_class_desc = 'oil_and_gas_infrastructure'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_oil_and_gas_infrastructure.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0,1,0]
    valid_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_turbine.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0,0,1]
    valid_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_other.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)
 
    return train_crops, train_filename, train_feature, train_class, valid_crops, valid_filename, valid_feature, valid_class

def add_dist2land_training_crops_20170829_binary_turbine(image_size):
    
    train_dir = '/home/ubuntu/data/sar/training_crops_20170829/train/'+image_size+'/'
    valid_dir = '/home/ubuntu/data/sar/training_crops_20170829/validate/'+image_size+'/'
    train_class = []           
    train_filename = []
    train_crops = []
    train_feature = []

    valid_class = []
    valid_filename = []
    valid_crops = []
    valid_feature = []

    train_class_desc = 'oil_and_gas_infrastructure'
    train_class_array = [0]
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_oil_and_gas_infrastructure.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [1]
    train_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_turbine.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0]
    train_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_other.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    valid_class_array = [0]
    valid_class_desc = 'oil_and_gas_infrastructure'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_oil_and_gas_infrastructure.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [1]
    valid_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_turbine.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0]
    valid_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_other.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)
 
    return train_crops, train_filename, train_feature, train_class, valid_crops, valid_filename, valid_feature, valid_class


def add_dist2land_training_crops_20170829_binary_oil_and_gas(image_size):
    
    train_dir = '/home/ubuntu/data/sar/training_crops_20170829/train/'+image_size+'/'
    valid_dir = '/home/ubuntu/data/sar/training_crops_20170829/validate/'+image_size+'/'
    train_class = []           
    train_filename = []
    train_crops = []
    train_feature = []

    valid_class = []
    valid_filename = []
    valid_crops = []
    valid_feature = []

    train_class_desc = 'oil_and_gas_infrastructure'
    train_class_array = [1]
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_oil_and_gas_infrastructure.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0]
    train_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_turbine.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0]
    train_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/train/distance_to_land/all_train_other.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.items():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    valid_class_array = [1]
    valid_class_desc = 'oil_and_gas_infrastructure'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_oil_and_gas_infrastructure.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0]
    valid_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_turbine.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0]
    valid_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/training_crops_20170829/validate/distance_to_land/all_validate_other.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.items():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)
 
    return train_crops, train_filename, train_feature, train_class, valid_crops, valid_filename, valid_feature, valid_class



def add_dist2land_experiment_crops_20170815():
    
    train_dir = '/home/ubuntu/data/sar/experiment_crops_20170815/train/240x240/'
    valid_dir = '/home/ubuntu/data/sar/experiment_crops_20170815/validate/240x240/'
    train_class = []           
    train_filename = []
    train_crops = []
    train_feature = []

    valid_class = []
    valid_filename = []
    valid_crops = []
    valid_feature = []

    train_class_desc = 'oil_and_gas_infrastructure'
    train_class_array = [1,0,0]
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/train/distance_to_land/experiments_train_oil_and_gas_infrastructure.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.iteritems():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0,1,0]
    train_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/train/distance_to_land/experiments_train_turbine.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.iteritems():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    train_class_array = [0,0,1]
    train_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/train/distance_to_land/experiments_train_other.json') as json_data:
        json_train_data = json.load(json_data)
        for id_, item in json_train_data.iteritems():
            fn = id_.replace('.tif', '.png')
            train_filename.append(fn)
            train_feature.append(item['distance to land'])
            train_class.append(train_class_array)
            file_path = train_dir + '/' + train_class_desc + '/' + fn
            img  = imread(file_path)
            train_crops.append(img)

    valid_class_array = [1,0,0]
    valid_class_desc = 'oil_and_gas_infrastructure'  
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/validate/distance_to_land/experiments_validate_oil_and_gas_infrastructure.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.iteritems():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0,1,0]
    valid_class_desc = 'turbine'  
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/validate/distance_to_land/experiments_validate_turbine.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.iteritems():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)

    valid_class_array = [0,0,1]
    valid_class_desc = 'other'  
    with open('/home/ubuntu/data/sar/experiment_crops_20170815/validate/distance_to_land/experiments_validate_other.json') as json_data:
        json_validation_data = json.load(json_data)
        for id_, item in json_validation_data.iteritems():
            fn = id_.replace('.tif', '.png')
            valid_filename.append(fn)
            valid_feature.append(item['distance to land'])
            valid_class.append(valid_class_array)
            file_path = valid_dir + '/' + valid_class_desc + '/' + fn
            img  = imread(file_path)
            valid_crops.append(img)
 
    return train_crops, train_filename, train_feature, train_class, valid_crops, valid_filename, valid_feature, valid_class


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