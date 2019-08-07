from .unet_model import VGGUnet
import cv2
from scipy.ndimage import label
import numpy as np
from .colours import colors

class Inference(object):
    def __init__(self,
                model,
                color,
                input_shape=(224, 224, 3)):
        self.model = model
        self.input_shape = input_shape
        self.color = color
    
    def max_image(self, image):
        width, height = image.shape[:2]
        count = np.unique(label(image)[0]).shape[0]
        ones = np.ones((width, height))
        labeled_image = []
        summa = np.zeros((width, height))
        labeled_image.append(np.zeros((width, height)))
        for i in range(count-2,-1,-1):
            summa = summa + (i+1)*labeled_image[count-2-i]
            labeled_image.append(np.clip(label(image)[0] - (i*ones + (i+2)*summa),0,1))
        maximum = 0
        max_idx = 0
        for i, image in enumerate(labeled_image):
            if np.sum(image) > maximum:
                maximum = np.sum(image)
                max_idx = i
        return labeled_image[max_idx]

    def get_color_mask(self, mask, input_shape):
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1
        bin_mask = np.zeros(input_shape)
        color_mask = np.zeros(input_shape)
        for i in range(3):
            color_mask[:,:,i] = mask*self.color[i]
            bin_mask[:,:,i] = mask
        return np.uint8(color_mask), np.uint8(bin_mask)
    
    def predict(self, im):
        test_im = cv2.resize(im, self.input_shape[:2])
        test_im = test_im.reshape((1, *self.input_shape))
        result = self.model.predict(test_im.astype('uint8')).reshape(self.input_shape[:2])
        result = cv2.resize(result, (im.shape[1], im.shape[0]))
        #image = self.max_image(result)
        image = result
        color_mask, mask = self.get_color_mask(image, im.shape)
        background = np.uint8(im)*np.uint8(1-mask)
        hair = np.uint8(im)*np.uint8(mask)
        new_image = np.uint8(np.clip(hair*0.75 + color_mask*0.25, 0, 255)) + background
        return new_image
