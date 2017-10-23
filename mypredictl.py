#coding:utf-8
import numpy as np
import cPickle
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
    X_test = np.transpose(cut_imgs ,(0, 3, 1, 2))
    X_test = X_test.astype('float32')
    X_test /= 255.
    return X_test

if __name__=='__main__':

    nb_classes = 3
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

    model_file='./my_model_architecture.json'
    weights_file='./my_model_weights.h5'

    json_file = open(model_file, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weights_file)
    # model = model_from_json('./my_model_architecture.json')
    # model.load_weights('./my_model_weights.h5')
    print('test after load')

    filenames = ['ht_001.jpg']

    for jpgfile in filenames:
        img_temp = mpimg.imread('PaperFigure_'+jpgfile )
        img = img_temp [:,:,0:3]
        cut_imgs=cut([50,50],img)
        X_test = testdata_x(cut_imgs)

        classes=model.predict_classes(X_test, batch_size = batch_size)
    print classes
    #print(classes.shape)
    # print(classes)

    # for i in range(classes.shape[0]):
    #     print(classes[i] )
    #     plt.imshow(cut_imgs[i])
    #     plt.show()
	
        # ratio=np.sum(classes,0)
        # sum = np.sum(ratio)
        # print(ratio/sum)
        # X = [1,2,3,4]
        # Y = ['papule', 'blackhead', 'normal', 'pustule']
        # plt.bar(X ,ratio/sum, tick_label =Y )
        # for x, y in zip(X, ratio/sum):
            # plt.text(x + 0.03, y + 0.001, '%.2f' % y, ha='center', va='bottom')
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
