#coding:utf-8

import os
import sys
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.patches as pt
import PIL.Image





def binary_test(dir,model_binary_file='./second_try_architecture_binary.json',weights_binary_file='./second_try_weights_binary.h5'):
	json_file_binary = open(model_binary_file, 'r')
	model_json_binary = json_file_binary.read()
	json_file_binary.close()
	model_binary = model_from_json(model_json_binary)
	model_binary.load_weights(weights_binary_file)
	
	im = mpimg.imread(dir)
	row_num = np.round((im.shape[0])/50)
	colomn_num = np.round((im.shape[1])/50)
	
	newim = np.empty((im.shape),dtype = 'uint8')
	newim = im[:,:,:]
	
	for i in range(0,row_num):
		for j in range(0,colomn_num):
			cutimage = np.empty((1,50,50,im.shape[2]),dtype = 'uint8')
			cutimage[0,:,:,:] = im[i*50:i*50+50,j*50:j*50+50,:]
			cut_temp = cutimage[:,:,:,0:3]
			class_binary = model_binary.predict(cut_temp)
			print class_binary
			
			if class_binary[0][0]<0.8:
				newim[i*50:i*50+50,j*50:j*50+50,:]=0
	
	mpimg.imsave('./aaa.jpg',newim)

if __name__=='__main__':
	if len(sys.argv) == 2:
	
		from keras.models import model_from_json
		dir = sys.argv[1]
		
		model_binary_file='./second_try_architecture_binary.json'
		weights_binary_file='./second_try_weights_binary.h5'
		
		json_file_binary = open(model_binary_file, 'r')
		model_json_binary = json_file_binary.read()
		json_file_binary.close()
		model_binary = model_from_json(model_json_binary)
		model_binary.load_weights(weights_binary_file)
		
		im = mpimg.imread(dir)
		row_num = np.round((im.shape[0])/50)
		colomn_num = np.round((im.shape[1])/50)
		
		newim = np.empty((im.shape),dtype = 'uint8')
		newim = im[:,:,:]
		
		for i in range(0,row_num):
			for j in range(0,colomn_num):
				cutimage = np.empty((1,50,50,im.shape[2]),dtype = 'uint8')
				cutimage[0,:,:,:] = im[i*50:i*50+50,j*50:j*50+50,:]
				cut_temp = cutimage[:,:,:,0:3]
				transimage = np.transpose(cut_temp ,(0, 3, 1, 2))
				class_binary = model_binary.predict(cut_temp)
				print class_binary
				
				if class_binary[0][0]<0.5:
					newim[i*50:i*50+50,j*50:j*50+50,:]=0
		
		mpimg.imsave('./aaa_second_try.jpg',newim)
		
	else:
		print 'Usage: python binarytest.py Image_Destination'