#coding:utf-8

import cPickle
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import h5py

batch_size = 32
nb_classes = 7
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 55, 55
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def def_model():

	model = Sequential()

	model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
	                        border_mode='valid',
	                        input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
  
        return model
	

if __name__ == '__main__':	
	
	load_data = open('./train.pkl','rb')
	data = cPickle.load(load_data)
	labels = cPickle.load(load_data)
	load_data.close()

        load_data_t = open('./test.pkl','rb')
        data_t = cPickle.load(load_data_t)
        labels_t = cPickle.load(load_data_t)
        load_data_t.close()

	# the data, shuffled and split between tran and test sets
	X_train,y_train = data,labels

	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	
	X_train = X_train.astype('float32')
	
	X_train /= 255.
	
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')

	X_test,y_test = data_t,labels_t
        X_test = X_test.astype('float32')
	X_test /= 255.
	

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	model = def_model()

	model.load_weights('my_model_weights.h5')

	classes = model.predict_classes(X_test, batch_size = batch_size)
        
        pos = 0
        neg = 0
        for i in range(len(classes)):
                ret = classes[i]
                #if Y_test[i][ret] == 1:
                if int(y_test[i]) == ret: 
                    pos += 1
                else:
                    neg += 1
        print "Total %d test samples, pos: %d, neg: %d, precision: %f%%"%(len(classes), pos, neg, 100 * float(pos)/(pos + neg))
        
        #score = model.evaluate(X_test, Y_test, batch_size = batch_size)
	#print('score',score)
    
	
	
	

