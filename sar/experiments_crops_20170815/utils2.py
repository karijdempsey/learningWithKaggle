# https://github.com/fastai/courses/blob/master/deeplearning2/utils2.py
import glob

from scipy.misc import imread


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))
    
def dir2list_of_numpy(data_dir, label):
    """
        label = [1,0,0] or  [0,1,0] or [0,0,1]
    """
    
    labels = []           
    filenames = []
    crops = []

    for png_path in glob.glob(data_dir+"/*.png"):
        img  = imread(png_path)
        crops.append(img)
        
        png_filename = png_path.split("/")[-1]
        filenames.append(png_filename)
        
        labels.append(label)
        
    return crops, labels, filenames
        
        

        

    
    
    
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