import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt 
import random as rnd 
from skimage.transform import resize
import random as rnd 
from sklearn.feature_extraction.image import extract_patches_2d,reconstruct_from_patches_2d
import os
import json


## import keras
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization,UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

def transfer_inception(feature_layer,input_shape):
    #mixed0: edges  % approx factor of 8 -3, 256 channels
    #mixed4: shapes % approx factor of 16 -2 768 channels
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=input_shape)
    input_layer = base_model.layers[0].input
    for layer in base_model.layers:
        layer.trainable = False
        if layer.name == feature_layer:
            feature_layer = layer.output
            #new_model = Model(inputs=[base_model.layers[0].input], outputs=[layer.output])
            break
    return input_layer,feature_layer


def build_model_on_inception(feature_layer,input_shape):
    input_layer,feature_layer = transfer_inception(feature_layer,input_shape)
    
    x = Flatten()(feature_layer)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(1,activation='sigmoid')(x)

    new_model = Model(inputs=[input_layer],outputs=[output_layer]) 
    return new_model

## build cnn model
input_shape = (256,256,3)
model = build_model_on_inception('mixed10',input_shape)
model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split

data_path = '../data/data_tennis/'

train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

validation_datagen = ImageDataGenerator()

def load_data():
	L = []
	X = []
	train_files = os.listdir(data_path+'other/')
	for i,fname in zip(range(len(train_files)),train_files):
		im = imread(data_path+'other/'+fname)
		X.append(resize(im,input_shape,mode='reflect'))
		L.append(0)
	train_files = os.listdir(data_path+'tennis+court/')
	for i,fname in zip(range(len(train_files)),train_files):
		im = imread(data_path+'tennis+court/'+fname)
		X.append(resize(im,input_shape,mode='reflect'))
		L.append(1)
	X = np.stack(X,axis=0)
	Y = np.array(L)
	return X,Y

print('Loading...')
X,Y = load_data()
print(str(X.shape[0])+' images found')

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3)
train_datagen.fit(X_train)
validation_datagen.fit(X_validation)


class_weights={}
class_weights[0]=np.sum(Y_train)/len(Y_train)
class_weights[1]=( len(Y_train)-np.sum(Y_train) )/len(Y_train)
print(class_weights)

## train model
fweights = './models/weights_inception.hdf5'
with tf.device('/device:GPU:0'):
	model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001),metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath=fweights, verbose=1, save_best_only=True)
	train_generator = train_datagen.flow(X_train,Y_train,batch_size=32)
	validation_generator = validation_datagen.flow(X_validation,Y_validation)
	history = model.fit_generator(train_generator,
			epochs=100,
			verbose=1,
			callbacks=[checkpointer],
			class_weight=class_weights,
			validation_data=validation_generator,
			workers=3)

print('model weights saved in ' + fweights)


history_loss = history.history['loss']
history_loss_val = history.history['val_loss']
history_acc = history.history['acc']
history_acc_val = history.history['val_acc']

import pickle
fhistory = './models/history_inception.pckl'
pickle_out = open(fhistory,"wb")
pickle.dump([history_loss,history_loss_val,history_acc,history_acc_val], pickle_out)
pickle_out.close()

print('loss and accuracy convergance history saved in ' + fhistory)


