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

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import itertools
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
# y = np.array([1,1,2,2])
# pred = np.array([0.1,0.4,0.35,0.8])
# fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
#
# print(fpr)
# print(tpr)
# print(thresholds)
#
# from sklearn.metrics import auc
# print(auc(fpr, tpr))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()
if __name__=='__main__':

    nb_classes = 7
    sum_classes = np.zeros((1, nb_classes))

    load_data = open('./train.pkl', 'rb')
    data = cPickle.load(load_data)
    labels = cPickle.load(load_data)
    load_data.close()
    print('datashape,labelsshape', data.shape, labels.shape)

    load_data_t = open('./test.pkl', 'rb')
    data_t = cPickle.load(load_data_t)
    labels_t = cPickle.load(load_data_t)
    load_data_t.close()
    print('datashape,labelsshape', data_t.shape, labels_t.shape)

    # the data, shuffled and split between train and test sets
    X_train, y_train = data, labels
    X_train = X_train.astype('float32')
    # X_train = X_train.transpose(0, 3, 1, 2)
    X_train /= 255.
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    X_test, y_test = data_t, labels_t
    X_test = X_test.astype('float32')
    # X_test = X_test.transpose(0, 3, 1, 2)
    X_test /= 255.

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model_binary_file = './top_binary_model_architecture.json'
    weights_binary_file = './top_binary_model_weights.h5'

    json_file_binary = open(model_binary_file, 'r')
    model_json_binary = json_file_binary.read()
    json_file_binary.close()
    model_binary = model_from_json(model_json_binary)
    model_binary.load_weights(weights_binary_file)

    model_seven_file = './top_seven_model_architecture.json'
    weights_seven_file = './top_seven_model_weights.h5'

    json_file_seven = open(model_seven_file, 'r')
    model_json_seven = json_file_seven.read()
    json_file_seven.close()
    model_seven = model_from_json(model_json_seven)
    model_seven.load_weights(weights_seven_file)

    model_VGG16 = applications.VGG16(include_top=False, weights='imagenet')

    my_model_binary_file = './my_model_architecture_binary.json'
    my_weights_binary_file = './my_model_weights_binary.h5'

    json_file_binary = open(my_model_binary_file, 'r')
    model_json_binary = json_file_binary.read()
    json_file_binary.close()
    my_model_binary = model_from_json(model_json_binary)
    my_model_binary.load_weights(my_weights_binary_file)

    my_model_seven_file = './my_model_architecture_seven.json'
    my_weights_seven_file = './my_model_weights_seven.h5'

    json_file_seven = open(my_model_seven_file, 'r')
    model_json_seven = json_file_seven.read()
    json_file_seven.close()
    my_model_seven = model_from_json(model_json_seven)
    my_model_seven.load_weights(my_weights_seven_file)

    print('test after load')
    # bottleneck_features = model_VGG16.predict(X_test)
    # #
    # class_binary = model_seven.predict_classes(bottleneck_features)
    class_binary = my_model_seven.predict_classes(X_test)
    # class_binary1 = np_utils.to_categorical(class_binary, nb_classes)
    # for i in range(class_binary.shape[0]):
    #     print(class_binary[i],y_test[i])
    myclasses = ['class0','class1','class2','class3','class4','class5','class6']
    y_true = y_test
    y_pred = class_binary
    print(classification_report(y_true, y_pred,target_names = myclasses))

    conf_mat = confusion_matrix(y_true, y_pred)
    print('confusion_matrix',conf_mat)

    plot_confusion_matrix(conf_mat,myclasses,True)
    plt.show()
    # print(Y_test ,classes )
    # print(class_binary1.shape)

        # print(class_binary1[i],class_binary[i],Y_test[i])

    # y_test = Y_test
    # y_score = class_binary
    # lw = 2
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # threshold_arr = dict()
    # for i in range(nb_classes):
    #     fpr[i], tpr[i], threshold_arr[i] = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # plt.figure()
    # # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i in range(nb_classes):
    #     plt.plot(fpr[i], tpr[i], lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.3f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve of seven-classifier based on our neural network')
    # plt.legend(loc="lower right")
    # plt.show()

    # for i in range(nb_classes):
    #     print('calss%d' % i)
    #
    #     fpr[i], tpr[i], threshold_arr[i] = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #     X = fpr[i]
    #     Y = tpr[i]
    #     max_index = np.argmax(Y - X)
    #     max_num = np.max(Y - X)
    #     # print('max_index', max_index)
    #     # print('J,max_num', max_num)
    #     threshold = threshold_arr[i][max_index]
    #     # print('threshold', threshold_arr[i][max_index])
    #     # print('fpr,tpr', (fpr[i][max_index], tpr[i][max_index]))
    #     #print('threshold_arr', threshold_arr)

    # for i in range(class_binary1.shape[0]):
   # print(y_score [:,i],y_test [:,i])
   #  for i in range(nb_classes):
   #      TP = 0.0
   #      FP = 0.0
   #      FN = 0.0
   #      TN = 0.0
   #      for y_s,y_t in zip(class_binary1[:,i],Y_test[:,i]):
   #          if (y_s == 1):
   #              if (y_t == 1):
   #                  TP += 1
   #              else:
   #                  FP += 1
   #          else:
   #              if (y_t == 1):
   #                  FN += 1
   #              else:
   #                  TN += 1
   #      #print(TP ,TN ,FN ,FP )
   #      # print('AUC',roc_auc[i])
   #      # print("Youden's index", max_num)
   #      # print('threshold', threshold_arr[i][max_index])
   #      print('calss%d' % i)
   #      print('Accuracy', (TP + TN) / (TP + FN + FP + TN))
   #      print('Sensitivity', TP / (TP + FN))
   #      print('Specificity', TN / (TN + FP))
   #      print(TP/(TP+FP))


