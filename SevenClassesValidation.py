# coding:utf-8

import os
import sys
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import matplotlib.patches as pt
import PIL.Image
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import model_from_json
import skimage.transform as st

def binary_test(dir, model_binary_file='./my_model_architecture_binary1.json',
                weights_binary_file='./my_model_weights_binary1.h5'):
    json_file_binary = open(model_binary_file, 'r')
    model_json_binary = json_file_binary.read()
    json_file_binary.close()
    model_binary = model_from_json(model_json_binary)
    model_binary.load_weights(weights_binary_file)

    im = mpimg.imread(dir)
    im = st.resize(im,[(im.shape[0]/50)*50, (im.shape[0]/50)*50, 3])
    print('qqqqqq',im.shape)
    row_num = np.round((im.shape[0]) / 50)
    colomn_num = np.round((im.shape[1]) / 50)

    newim = np.empty((im.shape), dtype='uint8')
    newim = im[:, :, :]

    for i in range(0, row_num):
        for j in range(0, colomn_num):
            cutimage = np.empty((1, 50, 50, im.shape[2]), dtype='uint8')
            cutimage[0, :, :, :] = im[i * 50:i * 50 + 50, j * 50:j * 50 + 50, :]
            cut_temp = cutimage[:, :, :, 0:3]
            class_binary = model_binary.predict(cut_temp)
            print class_binary

            if class_binary[0][0] < 0.8:
                newim[i * 50:i * 50 + 50, j * 50:j * 50 + 50, :] = 0

    mpimg.imsave('./aaa.jpg', newim)


def save_bottlebeck_features(data):
    datagen = ImageDataGenerator(rescale=1. / 255)
    nb_validation_samples = 1426
    batch_size = 16
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        data,
        target_size=(50, 50),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_validation_samples // batch_size)
    return bottleneck_features_train


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dir = sys.argv[1]

    sum_classes = np.zeros((1,7))

    model_binary_file = './models/new_top_binary_model_architecture.json'
    weights_binary_file = './models/new_top_binary_model_weights.h5'

    json_file_binary = open(model_binary_file, 'r')
    model_json_binary = json_file_binary.read()
    json_file_binary.close()
    model_binary = model_from_json(model_json_binary)
    model_binary.load_weights(weights_binary_file)

    model_seven_file = './models/my_model_architecture_seven_epoch100a.json'
    weights_seven_file = './models/my_model_weights_seven_epoch100a.h5'

    json_file_seven = open(model_seven_file, 'r')
    model_json_seven = json_file_seven.read()
    json_file_seven.close()
    model_seven = model_from_json(model_json_seven)
    model_seven.load_weights(weights_seven_file)

    model_VGG16 = applications.VGG16(include_top=False, weights='imagenet')

    # filedir = 'data/validation/skin'
    # filelist = os.listdir(filedir)

    newim = mpimg.imread(dir)
    # newim = PIL.Image.open(dir)
    im = np.empty([(newim.shape[0] / 50) * 50, (newim.shape[0] / 50) * 50, 3],dtype = 'uint8')

    # im = st.resize(newim, [(newim.shape[0] / 50) * 50, (newim.shape[0] / 50) * 50, 3])*255
    im = newim[0:(newim.shape[0] / 50) * 50, 0:(newim.shape[0] / 50) * 50, :]
    # im = PIL.Image.resize(newim,[(newim.shape[0] / 50) * 50, (newim.shape[0] / 50) * 50, 3])
    # im = newim.resize([(newim.shape[0] / 50) * 50, (newim.shape[0] / 50) * 50, 3])
    print('qqqqq',im,im.shape)
    row_num = np.round((im.shape[0]) / 50)
    colomn_num = np.round((im.shape[1]) / 50)

    newim = np.empty((im.shape), dtype='uint8')
    newim = im[:, :, :]

    for i in range(0, row_num):
        for j in range(0, colomn_num):
            cutimage = np.empty((1, 50, 50, im.shape[2]), dtype='uint8')
            cutimage[0, :, :, :] = im[i * 50:i * 50 + 50, j * 50:j * 50 + 50, :]
            cut_temp = cutimage[:, :, :, 0:3] * 1.0 / 255
            transimage = np.transpose(cut_temp, (0, 3, 1, 2))

            bottleneck_features = model_VGG16.predict(transimage)
            print(bottleneck_features.shape)
            class_binary = model_binary.predict(bottleneck_features)
            print class_binary
            # print class_binary

            if class_binary[0][1] < 0.5:
                newim[i * 50:i * 50 + 50, j * 50:j * 50 + 50, :] = 0
            else:
                # class_seven = model_seven.predict(bottleneck_features)
                # print(class_seven)
                # sum_classes = sum_classes + class_seven
                pass
    print('sum_classes',sum_classes)
    print('ratio',sum_classes/np.sum(sum_classes))

    mpimg.imsave('./test_file/output.jpg', newim)

