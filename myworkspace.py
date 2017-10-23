#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.patches as mpatches
from skimage import data,color,morphology,draw,measure, filters
import cv2
import numpy as np

#skin detect
def detect( img ):
    #img_temp = color.rgb2hsv(img)
    #img_temp = color.rgb2grey(img)

    img_temp = myrgb2ycbcr(img)

    #sizeimg_temp.shape
    map=np.zeros([img.shape[0],img.shape[1]])

    for i in range(img_temp.shape[0]):
        for j in range(img_temp.shape[1]):
            if ((img_temp[i,j,1]>=75)&(img_temp[i,j,1]<=135)&(img_temp[i,j,2]>=130)&(img_temp[i,j,2]<=180)):
            # if ((img_temp[i, j, 1] >= -20) & (img_temp[i, j, 1] <= 0) & (img_temp[i, j, 2] >= 10) & (
            #     img_temp[i, j, 2] <= 30)):
                map[i,j]=1

    # plt.figure()
    # plt.imshow(map)
    # plt.title('map')
    # plt.show()
    return map
#rgb2ycbcr
def myrgb2ycbcr(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Y = 0.299*R+ 0.587*G+ 0.114*B
    # Cb = -0.169*R -0.332*G +0.500*B
    # Cr = 0.500*R -0.419*G -0.081*B
    Y = 0.257 * R + 0.564 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
    img_ycbcr = np.zeros (img.shape)
    img_ycbcr[:, :, 0] = Y
    img_ycbcr[:, :, 1] = Cb
    img_ycbcr[:, :, 2] = Cr
    # print(img_ycbcr.shape)

    return img_ycbcr
#compensate
def compensate(img):
    bw = color.rgb2gray(img)
    len = bw.shape[0] * bw.shape[1]
    bw_arr = bw.reshape(len)
    h=-np.sort(-bw_arr )     #像素值排序
    len_5 = int(len*0.01)
    v=np.mean(h[0:len_5 ])#最亮前%5的像素平均值
    I_changed = np.zeros(img.shape)
    # I_changed = img*(1/v)                #按(255/v)系数补偿
    # I_changed = np.array(I_changed ,dtype = 'uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                I_changed [i,j,k] = img[i,j,k]*(1/v)
                if(I_changed [i,j,k]>255):
                    I_changed [i,j,k]=255
    I_changed = np.array(I_changed, dtype='uint8')


    #print(a)
    # print('hh',type(img),type(I_changed ))
    # print(len_5, v, 1/v)
    return I_changed
#morphology
def morphology_operation(img,struct_element=np.ones((15,15))):
    img_eroded = cv2.erode(img, struct_element)
    # 膨胀图像
    img_dilated = cv2.dilate(img_eroded, struct_element)
    img_mophology = img_dilated
    return img_mophology

def myplot(img,name=''):
    plt.figure()
    plt.imshow (img)
    plt.title (name)

# #生成二值测试图像
# img=mpimg.imread('area.jpg')
# print('test')
# # chull = morphology.convex_hull_image(~img)
# chull = morphology.convex_hull_object(~img,neighbors=8)
# #绘制轮廓
# print('test1')
# fig, axes = plt.subplots(1,2,figsize=(8,8))
# ax0, ax1= axes.ravel()
# ax0.imshow(img,plt.cm.gray)
# ax0.set_title('original image')
#
# ax1.imshow(chull,plt.cm.gray)
# ax1.set_title('convex_hull image')
# plt.show()

#定义了一个5×5的十字形结构元素,
#用结构元素与其覆盖的二值图像做“与”操作
#如果都为1，结果图像的该像素为1,否则为0
#腐蚀处理的结果是使原来的二值图像减小一圈。

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

if __name__=='__main__':
    #step1 读入图片
    img = mpimg.imread('baitou_test.jpg')
    # a = histeq(img)
    # plt.imshow()
    myplot(img,'origin')
    #step2.1光照补偿
    img_com = compensate(img)
    myplot(img_com,'img_compensate')
    #step2.2 skin detect
    map1 = detect(img)
    map2 = detect(img_com)
    myplot(map1,'map1')
    myplot(map2,'map2')

    #step2.3 morphology
    img_morphology1 = morphology_operation(map1)
    img_morphology2 = morphology_operation(map2)
    myplot(img_morphology1 ,'img_morphology1')
    myplot(img_morphology2 ,'img_morphology2')
    #step2.4 remove small objects
    img_temp1 = img_morphology1.astype('bool')
    area1 = morphology.remove_small_objects(~img_temp1, min_size=5000)#min_size=2000
    area1 = morphology.remove_small_objects(~area1, min_size=5000)  # min_size=2000
    area1 = ~area1
    #chull = morphology.convex_hull_object(~area, neighbors=8)
    #area = ~area
    #myplot(chull,'chull')
    #chull = filters.gaussian(~area, sigma=1)
    med1 = filters.median(~area1, morphology.disk(20))
    myplot(med1,'med1')
    print(med1)
    med1 = np.array((med1),dtype='uint8')
    result1 =np.empty((img.shape),dtype='uint8')
    result1[:,:,0] = med1/255 * img[:, :, 0]
    result1[:,:,1] = med1/255 * img[:, :, 1]
    result1[:,:,2] = med1/255 * img[:, :, 2]

    result1 = result1.astype('uint8')
    myplot(result1,'result1')

    img_temp2 = img_morphology2.astype('bool')
    area2 = morphology.remove_small_objects(~img_temp2, min_size=5000)  # min_size=2000
    area2 = morphology.remove_small_objects(~area2, min_size=5000)  # min_size=2000
    area2 = ~area2
    # chull = morphology.convex_hull_object(~area, neighbors=8)
    # area = ~area
    # myplot(chull,'chull')
    # chull = filters.gaussian(~area, sigma=1)
    med2 = filters.median(~area2, morphology.disk(20))
    myplot(med2, 'med2')
    print(med2)
    med2 = np.array((med2), dtype='uint8')
    result2 = np.empty((img.shape), dtype='uint8')
    result2[:, :, 0] = med2 / 255 * img[:, :, 0]
    result2[:, :, 1] = med2 / 255 * img[:, :, 1]
    result2[:, :, 2] = med2 / 255 * img[:, :, 2]

    result2 = result2.astype('uint8')
    myplot(result2, 'result2')
    #plt.show()
    plt.figure()
    plt.subplot(2,5,1)
    plt.imshow(img)
    plt.title('img')
    plt.subplot(2,5,6)
    plt.imshow(img_com)
    plt.title('img_com')
    plt.subplot(2, 5, 2)
    plt.imshow(map1)
    plt.title('map1')
    plt.subplot(2, 5, 7)
    plt.imshow(map2)
    plt.title('map2')
    plt.subplot(2, 5, 3)
    plt.imshow(img_morphology1)
    plt.title('img_morphology1')
    plt.subplot(2, 5, 8)
    plt.imshow(img_morphology2)
    plt.title('img_morphology2')
    plt.subplot(2, 5, 4)
    plt.imshow(med1 )
    plt.title('med1')
    plt.subplot(2, 5, 9)
    plt.imshow(med2)
    plt.title('med2')
    plt.subplot(2, 5, 5)
    plt.imshow(result1)
    plt.title('result1')
    plt.subplot(2, 5, 10)
    plt.imshow(result2)
    plt.title('result2')

    #mpimg.imsave('./med.jpg',med)
    #fig, ax=plt.subplots(1 ,2, figsize=(8,6) )
    plt.show()
    '''
    #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
    result = cv2.absdiff(dilated,eroded);
    dilated_temp=dilated.astype('bool')
    area= morphology.remove_small_objects(~dilated_temp , min_size=1)
    plt.imshow(img)

    plt.figure()
    plt.imshow (dilated)

    plt.figure()
    plt.imshow (~dilated)

    plt.figure()
    plt.imshow(~area)
    plt.show()

    areas = measure.label(area,connectivity=None)
    # plt.figure()
    fig,ax1= plt.subplots(1,1, figsize=(8, 6))

    ax1.imshow(areas)
    for region in measure.regionprops(areas):  # 循环得到每一个连通区域属性集

        # # 忽略小区域
        # if region.area < 100:
        #     continue

        # 绘制外包矩形
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)

    plt.show()
    # fig.tight_layout()
    # plt.show()

    # #取反
    # x=0;
    # y=0;
    # width=result.shape[0]
    # height=result.shape[1]
    # while x<width:
    # 	y=0
    # 	while y<height:
    # 		result[x][y]=255-result[x][y]
    # 		y=y+1;
    # 	x=x+1
    # cv2.imwrite("./eroded.jpg", eroded)
    # cv2.imwrite("./dilated.jpg", dilated)
    # cv2.imwrite("./result.jpg", result)
    #
    #
    #cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''