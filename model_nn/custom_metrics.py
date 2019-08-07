import keras
import numpy as np
from metrics import validate
from keras.callbacks import Callback

class CustomMetrics(keras.callbacks.Callback):

    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps


    def on_epoch_end(self, batch, epoch, logs={}):
        
        score_mean_iou = []
        score_mean_acc = []

        for batch_index in range(self.validation_steps):
            x_val, y_true = next(self.validation_generator)            
            y_pred = np.round(np.asarray(self.model.predict(x_val))).astype('uint8')
            acc, mean_acc, mean_iou = validate(y_pred[:,:,:,0], y_true[:,:,:,0], 2)
            score_mean_iou.append(mean_iou)
            score_mean_acc.append(mean_acc)
            
        print("--------------------------------------------------------")
        print("mean_iou for epoch is ", str(np.mean(score_mean_iou)))
        print("mean_acc for epoch is ", str(np.mean(score_mean_acc)))
        print("--------------------------------------------------------")
        return