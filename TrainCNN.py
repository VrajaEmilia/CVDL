import os
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
loadedImages = []
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

class DataLoader():
    def __init__(self):
        self.DATASET_PATH = './data/'
        self.GESTURES = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']
        self.load()

    def load(self):
        for i in range(0, len(self.GESTURES)):
            directory = self.DATASET_PATH + self.GESTURES[i]
            for filename in os.listdir(directory):
                imgPath = directory + '/' + filename
                image = cv2.imread(imgPath)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, (100, 120))
                loadedImages.append(gray_image)

#loads the data
#assigns 0-blank
#        1-fist
#        2-five
#        3-ok
#        4-thumbsdown
#        5-thumbsup
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
#
# batch = data_iterator.next()
# fig,ax = plt.subplots(ncols = 12, figsize=(20,20))
# for idx, img in enumerate(batch[0][:12]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# fig.show()

#scale the data
data = data.map(lambda x,y: (x/255,y))

# split the data
train_size = int(len(data)*0.6) + 2  #195 batches
val_size = int(len(data)*0.2)        #64 batches
test_size = int(len(data)*0.2)       #64 batches

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+test_size).take(test_size)