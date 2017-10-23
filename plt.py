#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

def histeq(image_array,image_bins=256):
    # 将图像矩阵转化成直方图数据，返回元组(频数，直方图区间坐标)
    image_array2,bins = np.histogram(image_array.flatten(),image_bins)
    # 计算直方图的累积函数
    cdf = image_array2.cumsum()
    # 将累积函数转化到区间[0,255]
    cdf = (255.0/cdf[-1])*cdf
    # 原图像矩阵利用累积函数进行转化，插值过程
    image2_array = np.interp(image_array.flatten(),bins[:-1],cdf)
    # 返回均衡化后的图像矩阵和累积函数
    return image2_array.reshape(image_array.shape),cdf
if __name__ == '__main__':

    filename = ['baitou_test.jpg','bt_091.jpg','ht_001.jpg','test0.jpg']
    j = 0
    for i in filename:
        j = j+1
        img = mpimg.imread(i)
        img_skin = mpimg.imread ('PaperFigure_'+i)
        plt.subplot (2,4,j)
        plt.axis('off')
        plt.imshow (img)
        plt.subplot (2,4,j+4)
        plt.axis('off')
        plt.imshow(img_skin )
    # for i in filename:
    #     img = mpimg.imread (i)
    #     hb = histeq (img)
    #     Hist_img = np.array(hb[0], dtype='uint8')
    #     mpimg.imsave ('Hist_'+i, Hist_img )

    plt.show()
