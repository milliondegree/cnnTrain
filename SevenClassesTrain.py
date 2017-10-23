# coding:utf-8
import numpy as np
import cPickle
import os
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import skimage.transform as st
np.random.seed(1337)  # for reproducibility
# dimensions of our images.
img_width, img_height = 50, 50
epochs = 100
batch_size = 16
nb_classes = 2

def save_bottlebeck_features(input,batch_size = 16):
    # build the VGG16 network
    model_VGG16 = applications.VGG16(include_top=False, weights='imagenet')
    # plot_model(model_VGG16, to_file = './models/model_VGG16.jpg', show_shapes = False, rankdir = 'LR')
    bottleneck_features_VGG16 = model_VGG16.predict(input, batch_size )

    # model_ResNet50 = applications.ResNet50(include_top=False, weights='imagenet')
    # bottleneck_features_ResNet50 = model_ResNet50.predict(input, batch_size )

    # model_InceptionV3 = applications.InceptionV3(include_top=False, weights='imagenet')
    # bottleneck_features_InceptionV3 = model_InceptionV3.predict(input, batch_size )

    # print(bottleneck_features_VGG16.shape, bottleneck_features_ResNet50.shape, bottleneck_features_InceptionV3.shape)
    # return bottleneck_features_VGG16, bottleneck_features_ResNet50, bottleneck_features_InceptionV3
    print(bottleneck_features_VGG16.shape)
    return bottleneck_features_VGG16


def train_top_model(train_data,train_labels,validation_data,validation_labels):
    # train_data = np.load(open('bottleneck_features_train.npy'))
    # train_labels_temp = np.array(
    #     [0] * (len(os.listdir(train_data_dir+'/notskin'))) + [1] * (len(os.listdir(train_data_dir+'/skin'))))
    #
    # validation_data = np.load(open('bottleneck_features_validation.npy'))
    # validation_labels_temp = np.array(
    #     [0] * (len(os.listdir(validation_data_dir+'/notskin'))) + [1] * (len(os.listdir(validation_data_dir+'/skin'))-2))
    # train_labels  = np_utils.to_categorical(train_labels_temp , nb_classes)
    # validation_labels = np_utils.to_categorical(validation_labels_temp , nb_classes)
    my_loss = 'mean_squared_error'
    # binary_crossentropy
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss=my_loss, metrics=[metrics.mae, metrics.categorical_accuracy])

    # plot_model(model, to_file = './models/my_model_VGG16.jpg', show_shapes = False, rankdir = 'LR')

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    json_string = model.to_json()
    open(top_model_architecture_path , 'w').write(json_string)
    print('save seven classifier model!')

if __name__ =='__main__':

    if nb_classes == 2:
        top_model_weights_path = './models/new_top_binary_model_weights.h5'
        top_model_architecture_path = './models/new_top_binary_model_architecture.json'
    if nb_classes == 7:
        top_model_weights_path = 'top_seven_model_weights.h5'
        top_model_architecture_path = 'top_seven_model_architecture.json'

    load_data = open('./pickled/train.pkl', 'rb')
    data = cPickle.load(load_data)
    labels = cPickle.load(load_data)
    load_data.close()
    print('datashape,labelsshape', data.shape, labels.shape)

    load_data_t = open('./pickled/test.pkl', 'rb')
    data_t = cPickle.load(load_data_t)
    labels_t = cPickle.load(load_data_t)
    load_data_t.close()
    print('datashape,labelsshape', data_t.shape, labels_t.shape)


    # the data, shuffled and split between train and test sets
    X_train, y_train = data, labels
    X_train = X_train.astype('float32')
    X_train = X_train.transpose(0, 3, 1, 2)
    X_train /= 255.
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    X_test, y_test = data_t, labels_t
    X_test = X_test.astype('float32')
    X_test = X_test.transpose(0, 3, 1, 2)
    X_test /= 255.

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)


    X_train_vector = save_bottlebeck_features(X_train )
    X_test_vector = save_bottlebeck_features(X_test )
    # save_bottlebeck_features(X_train )
    # save_bottlebeck_features(X_test )
    train_top_model(X_train_vector, Y_train, X_test_vector, Y_test)