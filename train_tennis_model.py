#!env/bin/python3
import sys

if len(sys.argv)==2:
  if sys.argv[1]=='-h':
    print('Usage:')
    print('   ' + sys.argv[0] + ' <path_to_data>')
    exit()
  else:
    model_type = 'hard'
    data_path = sys.argv[1]
else:
  print('Unexpected number of arguments.')
  print('Help: ' + sys.argv[0] + ' -h')
  exit()

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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from mymodels import weak_model, strong_model
from load_examples import load_data

import tensorflow as tf

## build cnn model
if model_type=='hard':
  model,input_shape = strong_model()
elif os.path.isdir(data_path + '/' + model_type):
  model,input_shape = weak_model() 
else:
  print(data_path + '/' + model_type + ' is cannot be found')
  print('Help: ' + sys.argv[0] + ' -h')
  exit()
model.summary()  
print('Loading...')
X,Y,F = load_data(data_path,model_type,input_shape)
print(str(X.shape[0])+' images found')

from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(
      rotation_range=180,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      vertical_flip=True)

validation_datagen = ImageDataGenerator()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3)
train_datagen.fit(X_train)
validation_datagen.fit(X_validation)

# setup class weights based on the unbalance
class_weights={}
class_weights[0]=np.sum(Y_train)/len(Y_train)
class_weights[1]=( len(Y_train)-np.sum(Y_train) )/len(Y_train)
print(class_weights)

## train model
fweights = './weights_' + model_type + '.hdf5'
fhistory = './history_' + model_type + '.pckl'

with tf.device('/device:GPU:0'):
  model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.001),metrics=['accuracy'])
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
pickle_out = open(fhistory,"wb")
pickle.dump([history_loss,history_loss_val,history_acc,history_acc_val], pickle_out)
pickle_out.close()

print('loss and accuracy convergance history saved in ' + fhistory)
