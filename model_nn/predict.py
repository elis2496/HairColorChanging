import keras
import numpy as np
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
import sys

from unet_model import VGGUnet
from tools import get_keys, get_data
from metrics import validate

def main():
    data_path = sys.argv[1]  
    weights_path = sys.argv[2]
 
    input_shape = (224, 224, 3)
    model = VGGUnet(input_shape)
    model.load_weights(weights_path)
    
    print('Loading_data')
    test_keys = get_keys('test', data_path)
    test_data = get_data(test_keys, (input_shape[0], input_shape[1]))
    img_test = np.array(test_data[0])
    
    y_true = np.array(test_data[1]).reshape((len(img_test),224,224)).astype('uint8')
    y_pred = model.predict(img_test, verbose=1).reshape((len(img_test),224,224)).astype('uint8')
    
    acc, mean_acc, mean_iou = validate(y_pred, y_true.astype('uint8'), 2)
    
    print("acc = %0.3f"%acc)
    print("mean_acc = %0.3f"%mean_acc)
    print("mean_iou = %0.3f"%mean_iou)
   
if __name__ == "__main__":
    main()
    
#example of testing: python predict.py ./data weights.h5py
    