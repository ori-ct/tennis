#!env/bin/python3
import sys

if len(sys.argv)==2:
  if sys.argv[1]=='-h':
    print('Usage:')
    print('   ' + sys.argv[0] + ' <model_type> <path_to_data>')
    print('     ' + '<model_type>:')
    print('       weak:   train againts the whole scene to get a weak classifier')
    print('       strong: train againts hard cases to get a strong classifier')
  else:
    print('Unexpected number of arguments')
    print('Help: ' + sys.argv[0] + ' -h')
  exit()
elif len(sys.argv)==3:
    model_type = sys.argv[1]
    data_path = sys.argv[2]
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
if model_type=='weak':
  model,input_shape = weak_model()
  model.summary()
  print('Loading...')
  X,Y,F = load_data(data_path,'pasadena',input_shape)
  print(str(X.shape[0])+' images found')
elif model_type=='strong':
  model,input_shape = strong_model()
  model.summary()  
  print('Loading...')
  X,Y,F = load_data(data_path,'hard',input_shape)
  print(str(X.shape[0])+' images found')
else:
  print(model_type + ' is an unknown type of model')
  print('Help: ' + sys.argv[0] + ' -h')
  exit()

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
if model_type=='weak':
  fweights = './weights_weak.hdf5'
  fhistory = './history_weak.pckl'
if model_type=='strong':
  fweights = './weights_strong.hdf5'
  fhistory = './history_strong.pckl'

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
