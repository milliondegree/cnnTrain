# coding:utf-8
import numpy as np
import cPickle
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
np.random.seed(1337)  # for reproducibility
# dimensions of our images.
img_width, img_height = 50, 50

top_model_weights_path = 'top_seven_model_weights.h5'
top_model_architecture_path = 'top_seven_model_architecture.json'
train_data_dir = 'data7classes/train'
validation_data_dir = 'data7classes/validation'
# nb_train_samples = 3568
# nb_validation_samples = 1426
epochs = 50
batch_size = 16
nb_classes = 7

def save_bottlebeck_features():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network

    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # train_bottleneck_features = model.predict_generator(train_generator, 3500)
    # test_bottleneck_features = model.predict_generator(validation_generator, 6378)
    result = model.fit_generator(train_generator,
        steps_per_epoch=3500 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=6378 // batch_size)
    print(result)

    # print(train_generator.classes.shape)
    # print(train_bottleneck_features.shape,test_bottleneck_features.shape)
    # return train_bottleneck_features, test_bottleneck_features


def train_top_model(train_data,train_labels,validation_data,validation_labels):
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    json_string = model.to_json()
    open(top_model_architecture_path , 'w').write(json_string)
    print('save seven classifier model!')

if __name__ =='__main__':
    # X_train_vector, X_test_vector = save_bottlebeck_features()
    save_bottlebeck_features()
    # print(x_train.shape,x_test.shape)

    # train_top_model(X_train_vector, Y_train, X_test_vector, Y_test)