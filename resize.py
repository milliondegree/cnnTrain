#coding:utf-8

import os
import PIL.Image as Image
import numpy as np

def fresize(theurl, destination):
	filelist = os.listdir(theurl);
	for filename in filelist:
		im = Image.open(theurl + filename)
		length, height = im.size
		out = im.resize((640,np.round(height*640/length)))
		out.save(destination + filename)
		
		
if __name__=='__main__':
	fresize('./原始样本/整图/囊肿zt/','./原始样本/整图/囊肿resize/')
	fresize('./原始样本/整图/白头zt/','./原始样本/整图/白头resize/')
	fresize('./原始样本/整图/结节zt/','./原始样本/整图/结节resize/')