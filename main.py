import cv2
import numpy as np
from keras.models import load_model

model = load_model("model.h5")

list_of_gestures = ['blank', 'ok', 'thumbsup', 'thumbsdown', 'fist', 'five']
image = cv2.imread('data/thumbsup/thumbsup1627.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (100, 120))

gray_image = gray_image.reshape(1, 100, 120, 1)

prediction = np.argmax(model.predict_on_batch(gray_image))

print(list_of_gestures[prediction])

