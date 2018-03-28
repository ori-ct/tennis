from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization,UpSampling2D,Activation
from keras.models import Model, Sequential

def strong_model():
  input_shape = (256,256,1)
  
  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  
  model.add(Flatten())

  model.add(Dense(128))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(1, activation='sigmoid'))

  return model,input_shape

def weak_model():
  input_shape = (256,256,1)
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(11, 11), strides=(1, 1),input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.2))

  model.add(Conv2D(32, (5, 5)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.2))

  #model.add(Conv2D(64, (3, 3)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.2))

  #model.add(Conv2D(64, (3, 3)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.2))
  
  model.add(Flatten())

  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))

  model.add(Dense(1, activation='sigmoid'))

  return model,input_shape
