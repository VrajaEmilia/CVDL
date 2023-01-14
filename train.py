import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import glob

model = Sequential()

# first conv layer
# input shape = (img_rows, img_cols, 1)
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 120, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))  # fully connected
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile('adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# load data
loaded_images = []

list_of_gestures = ['blank', 'ok', 'thumbsup', 'thumbsdown', 'fist', 'five']

for gesture in list_of_gestures:
    folder_path = 'data/' + gesture
    count = 0
    for file in glob.glob(folder_path + '/*'):
        file_path = folder_path + '/' + file[len(folder_path) + 1:]
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (100, 120))
        loaded_images.append(gray_image)
        count += 1
        if count >= 1000:
            break

outputVectors = []
for i in range(1, 1001):
    outputVectors.append([1, 0, 0, 0, 0, 0])

for i in range(1, 1001):
    outputVectors.append([0, 1, 0, 0, 0, 0])

for i in range(1, 1001):
    outputVectors.append([0, 0, 1, 0, 0, 0])

for i in range(1, 1001):
    outputVectors.append([0, 0, 0, 1, 0, 0])

for i in range(1, 1001):
    outputVectors.append([0, 0, 0, 0, 1, 0])

for i in range(1, 1001):
    outputVectors.append([0, 0, 0, 0, 0, 1])

X = np.asarray(loaded_images)
y = np.asarray(outputVectors)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0], 100, 120, 1)
X_test = X_test.reshape(X_test.shape[0], 100, 120, 1)

model.fit(X_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))

model.save("model.h5")
