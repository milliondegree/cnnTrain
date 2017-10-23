# coding:utf-8
import numpy as np
import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 50, 50

top_model_weights_path = 'bottleneck_fc_model_weights.h5'
top_model_architecture_path = 'bottleneck_fc_model_architecture.json'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3568
nb_validation_samples = 1426
epochs = 50
batch_size = 16
nb_classes = 2

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)
    print(bottleneck_features_train.shape)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)
    print(bottleneck_features_validation.shape)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels_temp = np.array(
        [0] * (len(os.listdir(train_data_dir+'/notskin'))) + [1] * (len(os.listdir(train_data_dir+'/skin'))))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels_temp = np.array(
        [0] * (len(os.listdir(validation_data_dir+'/notskin'))) + [1] * (len(os.listdir(validation_data_dir+'/skin'))-2))
    train_labels  = np_utils.to_categorical(train_labels_temp , nb_classes)
    validation_labels = np_utils.to_categorical(validation_labels_temp , nb_classes)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    json_string = model.to_json()
    open(top_model_architecture_path , 'w').write(json_string)
    print('save binary classifier model!')

save_bottlebeck_features()
train_top_model()