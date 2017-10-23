#coding:utf-8

import os
import sys
import numpy
#import cv2
import matplotlib.image as mpimg # mpimg 用于读取图片
import cPickle
import random
import matplotlib.pyplot as plt

def getDirLabel():
	root = './binarydata/BinaryClassifier'
	trainDir = []
	trainLabel = []
	testDir = []
	testLabel = []
	idx = -1
	for i in os.listdir(root):

		dirName = root + '/' + i
		idx += 1
		files = os.listdir(dirName)



		picture_list=list(range(0,len(files)))
		print len(picture_list)
		random.shuffle(picture_list)
		#cut = int(round(len(files) * 0.3))
		for j in range(len(picture_list)):
			fileName = dirName + '/' + files[picture_list[j]]
			if j<len(files)*0.3:
				testDir.append(fileName)
				testLabel.append(idx)
			else:
				trainDir.append(fileName)
				trainLabel.append(idx)
			# print(fileName,idx)
	# print len(trainDir)
	# print len(testDir)
	# print (trainDir, trainLabel)
	return trainDir, trainLabel, testDir, testLabel

def getDataLabels(directory, label):
	img_rows, img_cols = 50, 50
	num = len(directory)
	labels = numpy.empty(num)
	data = numpy.empty((num,img_cols ,img_rows ,3))
	idx = 0
	#image=cv2.imread(directory[0])
	#print('image_shape',image.shape)
	#cv2.imwrite('image.jpg',image)
	for i in directory:
		img = mpimg.imread(i)
		print i
		img.resize((img_rows, img_cols,3))#img_size=(r,c,3)
		#tmp = numpy.array(img.shape)
		data[idx, :, :, :] = img
		labels[idx] = label[idx]
		idx += 1
	#print('dshape lshape',data.shape,labels.shape)
	#image=cv2.imread(data[0,:,:,:])
	#print('image_shape',data[0].shape)
	#cv2.imwrite('image.jpg',data[0])
	return data, labels

def write2File(data, labels, name):
	writeFile = open(name, 'wb')
	cPickle.dump(data, writeFile, -1)
	cPickle.dump(labels, writeFile, -1)
	writeFile.close()

def main():
	trainDir, trainLabel, testDir, testLabel= getDirLabel()
	trainData, trainLabels = getDataLabels(trainDir, trainLabel)
	print "train data read"
	testData, testLabels = getDataLabels(testDir, testLabel)
	print "test data read"
	write2File(trainData, trainLabels, './pickled/train.pkl')
	print "train data write"
	write2File(testData, testLabels, './pickled/test.pkl')
	print "test data write"


main()


	
