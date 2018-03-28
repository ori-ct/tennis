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
input_shape = (256,256,1)
model,input_shape = weak_model()
model.summary()


from sklearn.model_selection import train_test_split

data_path = '../data/data_tennis/'


#model, input_shape = weak_model()
model, input_shape = strong_model()

model.summary()

X,Y,F = load_data_test(data_path,'pasadena',input_shape)

print(str(X.shape[0])+' images found')
## train model

#fweights = './weights_weak.hdf5'
fweights = './weights_strong.hdf5'

with tf.device('/device:GPU:0'):
	model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001),metrics=['accuracy'])
	model.load_weights(fweights)
	pred =model.predict(X)
pred=np.squeeze(pred)
results = [(f,s,l) for f,s,l in zip(F,pred,Y)]
df = pd.DataFrame(results,columns=['file_name','score','truth'])

#df.to_csv('./models/results_pasadena_gry_weak.csv',index=False)
df.to_csv('./models/results_pasadena_gry_strong.csv',index=False)

