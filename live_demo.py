import cv2
import numpy as np
from keras.models import load_model
bg = None

model=load_model('model.h5')
list_of_gestures = ['blank', 'ok', 'thumbsup', 'thumbsdown', 'fist', 'five']

if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    fps = int(camera.get(cv2.CAP_PROP_FPS))

    top, right, bottom, left = 10, 350, 225, 590
    calibrated = False
    k = 0

    while (True):
        (grabbed, frame) = camera.read()
        clone = frame.copy()
        clone = cv2.flip(clone, 1)
        frame = cv2.flip(frame, 1)
        cv2.rectangle(clone, (100, 100), (300, 300), (255, 255, 255), 0)
        handRegion = frame[100:300, 100:300]
        gray = cv2.cvtColor(handRegion, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if bg is None:
            cv2.putText(clone, 'press b when you have the background ready', (20, 45), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 1)
            cv2.putText(clone, 'or if you change the background at anytime', (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                        (255, 255, 255), 1)
        else:
            hand = cv2.absdiff(bg, gray)
            _, thresholded = cv2.threshold(hand, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            resizedHand = cv2.resize(thresholded, (100, 120))
            cv2.imshow("hand", thresholded)
            resizedHand = resizedHand.reshape(1, 100, 120, 1)
            prediction = model.predict_on_batch(resizedHand)
            predicted_class = list_of_gestures[np.argmax(prediction)]
            print(predicted_class)
            cv2.putText(clone, predicted_class, (20, 90), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                        (0, 0, 255), 1)

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        if keypress == ord("b"):
            bg = gray

    # free up memory
    camera.release()
    cv2.destroyAllWindows()