import traceback

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    try:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # draw a rectangle where to place the hand
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), 0)

        handRegion = frame[100:300, 100:300]

        # convert to different color spaces
        hsvImage = cv2.cvtColor(handRegion, cv2.COLOR_BGR2HSV)
        YCbCrImage = cv2.cvtColor(handRegion, cv2.COLOR_BGR2YCrCb)

        # define the threshold in HSV color space
        min_hsv = np.array([0, 20, 70], dtype=np.uint8)
        max_hsv = np.array([20, 255, 255], dtype=np.uint8)

        # define the threshold in YCrCb color space
        min_YCrCb = np.array([0, 132, 76], dtype=np.uint8)
        max_YCrCb = np.array([255, 172, 128], dtype=np.uint8)

        mask1 = cv2.inRange(hsvImage, min_hsv, max_hsv)
        mask2 = cv2.inRange(YCbCrImage, min_YCrCb, max_YCrCb)
        # apply AND to maximize results
        result = cv2.bitwise_and(mask1, mask2)

        # use dilate to fill dark spots
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.dilate(result, kernel, iterations=4)

        cv2.imshow('camera', frame)
        cv2.imshow('result', result)
    except:
        traceback.print_exc()
        break

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
