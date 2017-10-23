#coding:utf-8
import os
import sys
import numpy as np
import cPickle
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import PIL.Image
#import cv2
np.random.seed(1337)#for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils
import h5py
from keras.models import model_from_json

def cut(cut_size , img):
    img_size = img.shape
    num_imgs = np.round(img_size[0]/cut_size[0])*np.round(img_size[1]/cut_size[1])
    cut_imgs=np.empty((num_imgs,cut_size[0],cut_size[1],img_size[2]),dtype='uint8')
    k=0
    for i in range(0,img_size[0],cut_size[0]):
        for j in range(0,img_size[1],cut_size[1]):
            cut_imgs[k,:,:,:] = img[i:i+50,j:j+50,:]
            k=k+1
    #cut_imgs =np.empty((2,500,500,3),dtype='uint8')
    #cut_imgs[0,:,:,:] = img[:,:,:]
    return cut_imgs

def testdata_x(cut_imgs):
    X_test = cut_imgs
    X_test = X_test.astype('float32')
    X_test = (X_test - 128)/128
    return X_test

if __name__=='__main__':

    nb_classes = 2
    batch_size = 1

    # load_data_t = open('./test.pkl','rb')
    # data_t = cPickle.load(load_data_t)
    # labels_t = cPickle.load(load_data_t)
    # load_data_t.close()
    # print('datashape,labelsshape',data_t.shape,labels_t.shape)
    #
    # X_test, y_test = data_t, labels_t
    # X_test = X_test.transpose(0, 3, 1, 2)
    #
    # X_test = X_test.astype('float32')
    # X_test /= 255.
    #
    # # convert class vectors to binary class matrices
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    # #model = Sequential()

    model_file_binary = './my_model_architecture_binary.json'
    weights_file_binary = './my_model_weights_binary.h5'

    json_file_binary = open(model_file_binary, 'r')
    model_json_binary = json_file_binary.read()
    json_file_binary.close()
    model_binary = model_from_json(model_json_binary)
    model_binary.load_weights(weights_file_binary)

    model_file_seven = './my_model_architecture_seven.json'
    weights_file_seven = './my_model_weights_seven.h5'

    json_file_seven = open(model_file_seven, 'r')
    model_json_seven = json_file_seven.read()
    json_file_seven.close()
    model_seven = model_from_json(model_json_seven)
    model_seven.load_weights(weights_file_seven)
    # model = model_from_json('./my_model_architecture.json')
    # model.load_weights('./my_model_weights.h5')
    print('test after load')

    #filenames = ['baitou_test.jpg', 'bt_091.jpg', 'ht_001.jpg', 'test0.jpg']
    filenames = os.listdir('./origin/zhengtu/baitouzt')
    for jpgfile in filenames:
        # img_temp0 = mpimg.imread('PaperFigure_'+jpgfile )
        img_temp0 = mpimg.imread(jpgfile)
        #img_temp0 = np.array(img_temp0 )
        img = img_temp0 [:,:,0:3]
        cut_imgs = cut([50,50],img)

        X_test = testdata_x(cut_imgs)

        classes_binary = model_binary.predict(X_test)
        #print(classes_binary)


        map = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        k = 0
        #classes = model_binary.predict(X_test)
        for i in range(0, img.shape[0], 50):
            for j in range(0, img.shape[1], 50):
                if (classes_binary [k][0] > 0.2):
                    map[i:i + 50, j:j + 50] = np.ones((50, 50))
                k = k + 1
        #print k
        #print classes
        img_temp = np.empty(img.shape, dtype='uint8')
        img_temp[:, :, :] = img_temp0 [:, :, 0:3]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (map[i, j] == 0):
                    img_temp[i, j, :] = 0


        plt.imshow(img_temp)
        plt.figure()
        plt.imshow (img)
        plt.show()
        # ratio = np.sum(classes, 0)
        # print(ratio)
        # mpimg.imsave('PaperFigure_' + jpgfile, img_temp)



    # for i in range(classes.shape[0]):
    #     print(classes[i] )
    #     plt.imshow(cut_imgs[i])
    #     plt.show()

        ratio=np.sum(classes_binary,0)
        sum = np.sum(ratio)
        print(ratio/sum)

        # X = [1,2,3,4]
        # Y = ['papule', 'blackhead', 'normal', 'pustule']
        # plt.bar(X ,ratio/sum, tick_label =Y )
        # for x, y in zip(X, ratio/sum):
        #     plt.text(x + 0.03, y + 0.001, '%.2f' % y, ha='center', va='bottom')
        # plt.show()

    #print('test',classes=model.predict(X_test),'true',Y_test )

    # img = mpimg.imread('bt_091.jpg')
    # plt.imshow(img)
    #
    # #    img = PIL.Image.open(jpgfile)
    # #    print(img.shape, type(img))
    # cut_imgs = cut([50, 50], img)
    # #    print(cut_imgs.shape)
    # #    print(cut_imgs[0].shape)
    # # print(type(cut_imgs[99]))
    # plt.figure()
    # plt.imshow(cut_imgs[99])
    # plt.show()
