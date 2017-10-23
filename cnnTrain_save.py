# coding:utf-8

import cPickle
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras import metrics
# from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, plot_model

from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.advanced_activations import LeakyReLU

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp

import h5py
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    load_data = open('./pickled/train_seven.pkl', 'rb')
    data = cPickle.load(load_data)
    labels = cPickle.load(load_data)
    load_data.close()
    print('datashape,labelsshape', data.shape, labels.shape)

    load_data_t = open('./pickled/test_seven.pkl', 'rb')
    data_t = cPickle.load(load_data_t)
    labels_t = cPickle.load(load_data_t)
    load_data_t.close()
    print('datashape,labelsshape', data_t.shape, labels_t.shape)

    batch_size = 64
    nb_classes = 7
    nb_epoch = 100

    # input image dimensions
    img_rows, img_cols = 50, 50
    # number of convolutional filters to use
    # nb_filters = 32
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # the data, shuffled and split between train and test sets
    X_train, y_train = data, labels

    # X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    # X_train = X_train.transpose(0, 3, 1, 2)
    # X_train = X_train.astype('float32')
    #X_train = (X_train -128)/128
    X_train /= 255.
    # print(data.shape,labels.shape)
    # plt.figure(1)
    # plt.imshow(data[1])


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    X_test, y_test = data_t, labels_t
    # X_test = X_test.transpose(0, 3, 1, 2)

    X_test = X_test.astype('float32')
    #X_test = (X_test -128)/128
    X_test /= 255.

    # print('y_train',y_train)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)



    #origin model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape = (img_rows, img_cols, 3 ), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    # model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),data_format='channels_last'))
    # model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters*2, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters*2, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),data_format='channels_last'))
    # model.add(Dropout(0.5))
    

    model.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters*4, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D((1, 1), data_format='channels_last'))
    model.add(Conv2D(filters = nb_filters*4, kernel_size = (nb_conv, nb_conv), strides = 1,padding='valid',data_format='channels_last'))
    # model.add(BatchNormalization(axis = 0, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),data_format='channels_last'))


    model.add(Flatten())
    model.add(Dense(4096))
    # model.add(BatchNormalization(axis = 1, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    # model.add(BatchNormalization(axis = 1, momentum = 0.9, epsilon = 1e-5, center = True, scale = True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # model1 = Sequential()
    # model1.add(Flatten(input_shape=[None,64,10,14]))
    # model1.add(Dense(128))
    # model1.add(Activation('relu'))
    # model1.add(Dropout(0.5))
    #
    # model1.add(Dense(nb_classes))
    # model1.add(Activation('softmax'))
    plot_model(model, to_file='./paper_pictures/my_cnn_model_binary_all_2.jpg', show_shapes=True)
    # plot_model(model1, to_file='./paper_pictures/my_cnn_model_binary_classifier.jpg', show_shapes=True)
    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    # model.compile(optimizer='rmsprop',
    #               loss=my_loss, metrics=[metrics.mae, metrics.categorical_accuracy])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1,validation_data=(X_test,Y_test))



    # hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    #           verbose=1, validation_data=(X_test, Y_test), validation_split=0.2)
    #
    #
    #
    if(nb_classes == 2):
        model.save_weights('new_my_model_weights_binary.h5')
        json_string = model.to_json()
        open('new_my_model_architecture_binary.json', 'w').write(json_string)
        print('save binary classifier model!')

    elif(nb_classes == 7):
        model.save_weights('new_my_model_weights_seven.h5')
        json_string = model.to_json()
        open('new_my_model_architecture_seven.json', 'w').write(json_string)
        print('save seven classifier model!')
    else:
        print('error!')
