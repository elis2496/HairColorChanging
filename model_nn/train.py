import keras
import numpy as np
import os

import cv2
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
from keras.callbacks import Callback

from keras.backend.tensorflow_backend import set_session

from generator import Generator
from unet_model import VGGUnet
from custom_metrics import CustomMetrics
from tools import get_keys, get_data
from metrics import validate
import sys

def main():
    
    data_path = sys.argv[1] 
    weights_path = sys.argv[2]
    
    input_shape = (224, 224, 3)

    train_keys = get_keys('train', data_path)
    val_keys = get_keys('validation', data_path)

    nb_epoch = 40
    batch_size = 10

    gen = Generator(batch_size,  
                    train_keys, 
                    val_keys,
                    (input_shape[0], input_shape[1]))

    metrics = CustomMetrics(gen.generate(False), gen.val_batches)
    
    model = VGGUnet(input_shape)
    
    history = model.fit_generator(gen.generate(True), gen.train_batches,
                                  nb_epoch, verbose=1,
                                  callbacks = [metrics],
                                  nb_worker=1)
    model.save_weights(weights_path)
    print("Done")
    
if __name__ == "__main__":
    main()
    
#example of testing: python train.py ./data new_weights.h5py
