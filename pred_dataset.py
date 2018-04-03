#!env/bin/python3
import sys

if len(sys.argv)==2:
  if sys.argv[1]=='-h':
    print('Usage:')
    print('   ' + sys.argv[0] + ' <path_to_scene> <output_file>')
  else:
    print('Unexpected number of arguments')
    print('Help: ' + sys.argv[0] + ' -h')
  exit()
elif len(sys.argv)==3:
    data_path = sys.argv[1]
    output_file = sys.argv[2]
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
import tensorflow as tf

from mymodels import weak_model,strong_model
from load_examples import load_data_test

## build cnn model
model,input_shape = weak_model()
model.summary()

from sklearn.model_selection import train_test_split

model, input_shape = weak_model()
model.summary()
X,Y,F = load_data_test(data_path,input_shape)
print(str(X.shape[0])+' images found')
fweights = './weights_weak.hdf5'
with tf.device('/device:GPU:0'):
	model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001),metrics=['accuracy'])
	model.load_weights(fweights)
	predw =model.predict(X)
predw=np.squeeze(predw)

model, input_shape = strong_model()
model.summary()
X,Y,F = load_data_test(data_path,input_shape)
print(str(X.shape[0])+' images found')
fweights = './weights_strong.hdf5'
with tf.device('/device:GPU:0'):
	model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001),metrics=['accuracy'])
	model.load_weights(fweights)
	preds =model.predict(X)
preds=np.squeeze(preds)


results = [(f,sw,ss,l) for f,sw,ss,l in zip(F,predw,preds,Y)]
df = pd.DataFrame(results,columns=['file_name','weak_score','strong_score','truth'])

#df.to_csv('./models/results_pasadena_gry_weak.csv',index=False)
df.to_csv(output_file,index=False)

