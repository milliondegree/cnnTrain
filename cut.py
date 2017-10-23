#coding:utf-8
import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.patches as pt
import PIL.Image

import myworkspace as workspace
import randomstr as rand

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils
import h5py
from keras.models import model_from_json



np.random.seed(1337)#for reproducibility

def cut(cut_size , img):
	img_size = img.shape
    #num_imgs = np.round(img_size[0]/cut_size[0])*np.round(img_size[1]/cut_size[1])
	row_num = np.round(img_size[0]/cut_size[0])-1
	colomn_num = np.round(img_size[1]/cut_size[1])-1
	num_imgs = row_num * colomn_num
	cut_imgs=np.empty((num_imgs,cut_size[0],cut_size[1],img_size[2]),dtype='uint8')
	k=0

	for i in range(0,row_num-1):
		for j in range(0,colomn_num-1):
			cut_imgs[k,:,:,:] = img[i*50:i*50+50,j*50:j*50+50,:]
			k=k+1
    #cut_imgs =np.empty((2,500,500,3),dtype='uint8')
    #cut_imgs[0,:,:,:] = img[:,:,:]
	return cut_imgs
	
def testdata_x(cut_imgs):
    X_test = np.transpose(cut_imgs ,(0, 3, 1, 2))
    X_test = X_test.astype('float32')
    X_test /= 255.
    return X_test
	
def find_max_index(list):
	length = len(list)
	index=0
	for i in range(0,length-1):
		if list[i] == max(list):
			index = i
	return index
	
def cut_resized(dir,dir_cut,dir_tag,index):
	filelist=os.listdir(dir)
	for filename in filelist:
		print filename
		imagetemp = mpimg.imread(dir+filename)
		img_size = imagetemp.shape
		row_num = np.round((img_size[0]-50)/20)
		colomn_num = np.round((img_size[1]-50)/20)
		
		n = 0
		m = 0
		newimage = np.empty(imagetemp.shape,dtype = 'uint8')
		newimage = imagetemp[:,:,:]
		
		for i in range(0,row_num):
			for j in range(0,colomn_num):
				cutimage = np.empty((1,50,50,img_size[2]),dtype = 'uint8')
				cutimage[0,:,:,:] = imagetemp[i*20:i*20+50,j*20:j*20+50,:]
				cut_temp = cutimage[:,:,:,0:3]
				class_binary = model_binary.predict(cut_temp)
				print class_binary
				
				if class_binary[0][0]<0.8:
					continue
				else:
					class_seven = model_seven.predict(cut_temp)
					print class_seven
					if class_seven[0][index]>0.206:
						mpimg.imsave(dir_cut + rand.randstr(20) +'.jpg',cutimage[0])
						print 'the '+np.str(n)+'th image saved'
						newimage[i*20+20:i*20+30,j*20+20:j*20+30,:]=0
						n = n+1
					elif class_seven[0][2]>0.8:
						mpimg.imsave('./原始样本/整图/正常_640/' + rand.randstr(20) + '.jpg',cutimage[0])
						m = m+1
		
		mpimg.imsave(dir_tag + filename, newimage)
	
	
def cut_binary_and_normal(dir,dir_cut):
	filelist=os.listdir(dir)
	for filename in filelist:
		print filename
		imagetemp = mpimg.imread(dir+filename)
		img_size = imagetemp.shape
		row_num = np.round((img_size[0]-50)/40)
		colomn_num = np.round((img_size[1]-50)/40)
		
		m = 0
		for i in range(0,row_num):
			for j in range(0,colomn_num):
				cutimage = np.empty((1,50,50,img_size[2]),dtype = 'uint8')
				cutimage[0,:,:,:] = imagetemp[i*40:i*40+50,j*40:j*40+50,:]
				cut_temp = cutimage[:,:,:,0:3]
				class_binary = model_binary.predict(cut_temp)
				print class_binary
				
				if class_binary[0][0]>0.5:
					continue
				else:
					class_seven = model_seven.predict(cut_temp)
					print class_seven
					if class_seven[0][2]>0.8:
						m = m+1
						if m == 10:
							mpimg.imsave('./原始样本/整图/正常_cut/' + rand.randstr(20) + '.jpg',cutimage[0])
							m = 0
					else:
						mpimg.imsave(dir_cut + rand.randstr(20) +'.jpg',cutimage[0])
	
	
	
def cut_single_img(dir,filename,dir_cut,dir_tag,index):
	imagetemp = mpimg.imread(dir+filename)
	img_size = imagetemp.shape
	row_num = np.round((img_size[0]-50)/30)
	colomn_num = np.round((img_size[1]-50)/30)
	
	n = 0
	# m = 0
	newimage = np.empty(imagetemp.shape,dtype = 'uint8')
	newimage = imagetemp[:,:,:]
	
	for i in range(0,row_num):
		for j in range(0,colomn_num):
			cutimage = np.empty((1,50,50,img_size[2]),dtype = 'uint8')
			cutimage[0,:,:,:] = imagetemp[i*30:i*30+50,j*30:j*30+50,:]
			cut_temp = cutimage[:,:,:,0:3]
			class_binary = model_binary.predict(cut_temp)
			print class_binary
			
			if class_binary[0][0]<0.8:
				continue
			else:
				YCrCbimg = workspace.myrgb2ycbcr(cut_temp[0])
				YCrCbimg.astype('float32')
				Ysum = 0
				for m in range(0,YCrCbimg.shape[0]-1):
					for n in range(0,YCrCbimg.shape[1]-1):
						Ysum = Ysum + YCrCbimg[m,n,0]
				Yave = sum / (YCrCbimg.shape[0] * YCrCbimg.shape[1])
				
				
				class_seven = model_seven.predict(cut_temp)
				print class_seven
				if class_seven[0][index]>0.206:
					mpimg.imsave(dir_cut + filename + '_' + np.str(n)+'.jpg',cutimage[0])
					print 'the '+np.str(n)+'th image saved'
					newimage[i*30+20:i*30+30,j*30+20:j*30+30,:]=0
					n = n+1
				# elif class_seven[0][2]>0.8:
					# mpimg.imsave('./原始样本/整图/正常from_nongbao/' + filename + '_' + np.str(m)+'.jpg',cutimage[0])
					# m = m+1
		
	mpimg.imsave(dir_tag + filename, newimage)
	
	
def print_Y(dir,filename):

	im = mpimg.imread(dir + filename)
	row_num = np.round((im.shape[0]-50)/30)
	colomn_num = np.round((im.shape[1]-50)/30)
	
	Ylist = np.empty((row_num,colomn_num))
	for i in range(0,row_num):
		for j in range(0,colomn_num):
			cutimage = np.empty((1,50,50,im.shape[2]),dtype = 'uint8')
			cutimage[0,:,:,:] = im[i*30:i*30+50,j*30:j*30+50,:]
			cut_temp = cutimage[:,:,:,0:3]
			
			YCrCbimg = workspace.myrgb2ycbcr(cut_temp[0])
			YCrCbimg.astype('float32')
			Ysum = 0.0
			for m in range(0,YCrCbimg.shape[0]):
				for n in range(0,YCrCbimg.shape[1]):
					Ysum = Ysum + YCrCbimg[n,m,0]
			Yave = Ysum / (YCrCbimg.shape[0] * YCrCbimg.shape[1])
			Ylist[i,j] = Yave
	
	np.savetxt('./aaa.txt',Ylist)
	
	
# model_binary_file='./my_model_architecture_binary.json'
# weights_binary_file='./my_model_weights_binary.h5'
# model_seven_file = './my_model_architecture_seven.json'
# weights_seven_file = './my_model_weights_seven.h5'

# json_file_binary = open(model_binary_file, 'r')
# model_json_binary = json_file_binary.read()
# json_file_binary.close()
# model_binary = model_from_json(model_json_binary)
# model_binary.load_weights(weights_binary_file)

# json_file_seven = open(model_seven_file,'r')
# model_json_seven = json_file_seven.read()
# json_file_seven.close()
# model_seven = model_from_json(model_json_seven)
# model_seven.load_weights(weights_seven_file)
	
if __name__=='__main__':

	model_binary_file='./second_try_architecture_binary.json'
	weights_binary_file='./second_try_weights_binary.h5'
	model_seven_file = './my_model_architecture_seven.json'
	weights_seven_file = './my_model_weights_seven.h5'
	
	json_file_binary = open(model_binary_file, 'r')
	model_json_binary = json_file_binary.read()
	json_file_binary.close()
	model_binary = model_from_json(model_json_binary)
	model_binary.load_weights(weights_binary_file)
	
	json_file_seven = open(model_seven_file,'r')
	model_json_seven = json_file_seven.read()
	json_file_seven.close()
	model_seven = model_from_json(model_json_seven)
	model_seven.load_weights(weights_seven_file)
	
	cut_binary_and_normal('./原始样本/整图/白头resize/','./原始样本/整图/白头cut/')
	cut_binary_and_normal('./原始样本/整图/囊肿resize/','./原始样本/整图/囊肿cut/')
	cut_binary_and_normal('./原始样本/整图/结节resize/','./原始样本/整图/结节cut/')

