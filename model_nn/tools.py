import cv2
import numpy as np
import os
from sklearn.utils import shuffle
from tqdm import tqdm

def get_rgb(data_path , object_name = 'hair'):
    """
    Function for obtaining pixel values(rgb) of a segmented object
    :param data_path - path to directory containing file with colormap
    :param object_name - name of segmented object
    :return rgb
    """
    colormap_file = open(os.path.join(data_path,'colormap.txt'))
    flag = True
    while flag:
        line = colormap_file.readline().split(' ')
        if line[0] == object_name:
            rgb = [np.round(int(line[1])/255.), np.round(int(line[2])/255.), np.round(int(line[3])/255.)]
            flag = False
    return rgb  

def get_mask(rgb_values, mask_path):
    """
     Function to get binary mask using pixel values of a segmented object
     :param mask_path - path to mask
     :param rgb_values - pixel values of a segmented object
     :return mask 
    """
    mask = np.round(cv2.imread(mask_path)[...,::-1]/255.)
    bin_mask = np.all(mask == rgb_values, axis=2)
    mask[bin_mask] = [1, 1, 1]
    mask[~bin_mask] = [0, 0, 0]
    return mask[:,:,0]

def get_keys(mode, path):
    """
    Function to get full path to images(keys)
    :param mode - "train", "test" or "validate"
    :param path - path to directory with data
    :return keys
    """    
    data_real = os.path.join(path, 'real')
    data_synt = os.path.join(path, 'synt')
    batch_size = 10
    keys = [os.path.join(data_real, 'masks', mode, keys) for keys in os.listdir(os.path.join(data_real, 'masks', mode))]
    synt_keys = [os.path.join(data_synt, 'masks',mode ,keys) for keys in os.listdir(os.path.join(data_synt, 'masks', mode))]
    keys.extend(synt_keys)
    keys = shuffle(keys)
    return keys


def get_data(keys, image_size):
    """    
    Function for getting data - images and masks of object
    :param keys - full path to images
    :param image_size - size of image
    :return imgs, masks
    """      
    imgs = []
    masks = []
    shuffle(keys)
    for key in tqdm(keys):
        try:
            img = cv2.imread(key.replace('masks','images')).astype('float32')
        except:
            img = cv2.imread(key.replace('masks','images').replace('..','.')).astype('float32')
        rgb_values = get_rgb(key[:key.find('masks')])
        mask = get_mask(rgb_values,key).astype('float32')
        img = cv2.resize(img, image_size).astype('float32')[...,::-1]
        mask = cv2.resize(mask, image_size).astype('float32')
        imgs.append((img/255.).reshape(image_size[0],image_size[1],3))
        mask = mask.reshape(image_size[0],image_size[1],1)
        masks.append(np.round(mask))
    return imgs, masks
