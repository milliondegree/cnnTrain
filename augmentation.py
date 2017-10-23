from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')
dir = './binarydata/BinaryClassifier/NotSkinArea'
imglist = os.listdir(dir)
for imgname in imglist:
    img = load_img(dir+'/'+imgname)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='./binarydata/ns_aug1/', save_prefix='notskin', save_format='jpg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely