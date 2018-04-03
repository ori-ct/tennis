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

def load_data(data_path,negative_class_name,input_shape):
	L = []
	X = []
	F = []
	train_files = os.listdir(data_path + '/' + negative_class_name + '/')
	for i,fname in zip(range(len(train_files)),train_files):
		im = imread(data_path+'/'+negative_class_name + '/'+fname)
		im = resize(im,(input_shape[0],input_shape[1],3),mode='reflect')
		X.append(np.squeeze(im[:,:,0]))
		X.append(np.squeeze(im[:,:,1]))
		X.append(np.squeeze(im[:,:,2]))
		L.append(0)
		L.append(0)
		L.append(0)
		F.append(fname)
		F.append(fname)
		F.append(fname)
	train_files = os.listdir(data_path+'/tennis+court/')
	for i,fname in zip(range(len(train_files)),train_files):
		im = imread(data_path+'tennis+court/'+fname)
		im = resize(im,(input_shape[0],input_shape[1],3),mode='reflect')
		X.append(np.squeeze(im[:,:,0]))
		X.append(np.squeeze(im[:,:,1]))
		X.append(np.squeeze(im[:,:,2]))
		L.append(1)
		L.append(1)
		L.append(1)
		F.append(fname)
		F.append(fname)
		F.append(fname)
	X = np.stack(X,axis=0)
	Y = np.array(L)
	X = np.expand_dims(X,axis=-1)
	return X,Y,F

def load_data_test(data_path,input_shape):
	L = []
	X = []
	F = []
	train_files = os.listdir(data_path)# + '/' + negative_class_name + '/')
	for i,fname in zip(range(len(train_files)),train_files):
		im = imread(data_path+'/'+negative_class_name + '/'+fname)
		im = resize(im,(input_shape[0],input_shape[1],3),mode='reflect')
		X.append(np.squeeze(np.mean(im,axis=-1)))
		L.append(0)
		F.append(fname)
	X = np.stack(X,axis=0)
	Y = np.array(L)
	X = np.expand_dims(X,axis=-1)
	return X,Y,F
