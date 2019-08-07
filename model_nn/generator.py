import numpy as np
import os
import cv2
from tools import get_rgb, get_mask
from sklearn.utils import shuffle
import random

class Generator(object):    
    def __init__(self,
                 batch_size, 
                 train_keys,  
                 val_keys, 
                 image_size,
                 saturation_var=0.5,
                 contrast_var=0.5,
                 brightness_var=0.5,                 
                 lighting_var=0.5):
        
        self.batch_size = batch_size
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)//self.batch_size
        self.val_batches = len(val_keys)//self.batch_size
        self.image_size = image_size
        self.saturation_var = saturation_var
        self.contrast_var = contrast_var
        self.brightness_var = brightness_var
        self.lighting_var = lighting_var
            
            
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_var
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
        

    def generate(self, train=True):
        
        while True:
            
            imgs = []
            masks = []
            
            if train:
                keys = self.train_keys
            else:
                keys = self.val_keys
                
            shuffle(keys)
            
            for key in keys:
                
                img = cv2.imread(key.replace('masks','images')).astype('float32')
                rgb_values = get_rgb(key[:key.find('masks')])
                mask = get_mask(rgb_values,key).astype('float32')
                
                if train:
                    
                    if random.random() < self.saturation_var:
                        img = self.saturation(img)
                    if random.random() < self.contrast_var:
                        img = self.contrast(img)
                    if random.random() < self.brightness_var:
                        img = self.brightness(img)
                    if random.random() < self.lighting_var:
                        img = self.lighting(img)
                

                img = cv2.resize(img, self.image_size).astype('float32')[...,::-1]
                mask = cv2.resize(mask, self.image_size).astype('float32')
                
                imgs.append((img/255.).reshape(self.image_size[0],self.image_size[1],3))
                mask = mask.reshape(self.image_size[0],self.image_size[1],1)
                masks.append(mask)
                
                if len(imgs) == self.batch_size:
                    
                    tmp_imgs = np.array(imgs)
                    tmp_masks = np.array(masks)
                    imgs = []
                    masks = []
                    yield tmp_imgs.reshape((self.batch_size,self.image_size[0],self.image_size[1],3)), tmp_masks.reshape((self.batch_size,self.image_size[0],self.image_size[1],1))