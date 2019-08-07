import keras
import numpy as np
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
import sys
import cv2

from unet_model import VGGUnet
from metrics import validate

def model(input_image):
    
    input_shape = (224, 224, 3)
    input_image = cv2.resize(input_image, (input_shape[0], input_shape[1])).astype('float32')
    input_image = input_image/255.
    weights_path = 'weights.h5py'
    model_vggunet = VGGUnet(input_shape)
    model_vggunet.load_weights(weights_path)
    segmentation_mask = model_vggunet.predict(input_image.reshape((1,224,224,3))).astype('uint8')
    
    return segmentation_mask

