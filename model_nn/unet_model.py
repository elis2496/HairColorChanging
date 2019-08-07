import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K



VGG_Weights_path = 'vgg16.h5py'

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

def VGGUnet(input_shape, mode=None):
    
    inputs = Input(shape = input_shape)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    x = Flatten(name='flatten')(pool5)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='softmax', name='predictions')(x)

    vgg  = Model(inputs, x)
    if mode != 'inference':
        try:
            vgg.load_weights(VGG_Weights_path)
        except:
            print('Loading_VGG16_weights')
            from keras.applications.vgg16 import VGG16
            base_model = VGG16(weights='imagenet')
            base_model.save_weights(VGG_Weights_path)
            print('Loading_done')
            vgg.load_weights(VGG_Weights_path)

    zp = ZeroPadding2D((1,1))(conv4)
    conv6 = Conv2D(512, (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv6)

    up7 = UpSampling2D( (2,2))(bn)
    concat = concatenate([up7, conv3], axis=3) 
    zp = ZeroPadding2D( (1,1))(concat)
    conv7 = Conv2D( 256, (3, 3))(zp)
    bn = BatchNormalization()(conv7)

    up8 = UpSampling2D( (2,2))(bn)
    concat = concatenate([up8, conv2], axis=3) 
    zp = ZeroPadding2D((1,1))(concat)
    conv8 = Conv2D( 128 , (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv8)

    up9 = UpSampling2D((2,2))(bn)
    concat = concatenate([up9, conv1], axis=3) 
    zp = ZeroPadding2D((1,1))(concat)
    conv9 = Conv2D( 64 , (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv9)
    
    conv10 =  Conv2D( 1 , (1, 1) , activation='sigmoid', padding='same')(bn)
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

def Unet(input_shape):
    
    inputs = Input(shape = input_shape)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

    zp = ZeroPadding2D((1,1))(conv4)
    conv6 = Conv2D(512, (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv6)

    up7 = UpSampling2D( (2,2))(bn)
    concat = concatenate([up7, conv3], axis=3) 
    zp = ZeroPadding2D( (1,1))(concat)
    conv7 = Conv2D( 256, (3, 3))(zp)
    bn = BatchNormalization()(conv7)

    up8 = UpSampling2D( (2,2))(bn)
    concat = concatenate([up8, conv2], axis=3) 
    zp = ZeroPadding2D((1,1))(concat)
    conv8 = Conv2D( 128 , (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv8)

    up9 = UpSampling2D((2,2))(bn)
    concat = concatenate([up9, conv1], axis=3) 
    zp = ZeroPadding2D((1,1))(concat)
    conv9 = Conv2D( 64 , (3, 3), padding='valid')(zp)
    bn = BatchNormalization()(conv9)
    
    conv10 =  Conv2D( 1 , (1, 1) , activation='sigmoid', padding='same')(bn)
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model