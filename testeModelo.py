import numpy as np
import cv2
from keras.models import load_model

cap = cv2.VideoCapture(0)

model = load_model('modelo.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return '20 KM/H'
    elif classNo == 1:
        return '30 KM/H'
    elif classNo == 2:
        return '50 KM/H'
    elif classNo == 3:
        return '60 KM/H'
    elif classNo == 4:
        return '70 KM/H'
    elif classNo == 5:
        return '80 KM/H'
    elif classNo == 6:
        return '100 KM/H'
    elif classNo == 7:
        return '120 KM/H'

while True:
    success, imgOrignal = cap.read()

    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    indexVal = np.argmax(predictions)
    probabilityValue = np.amax(predictions)
    print(indexVal,probabilityValue)

    cv2.putText(imgOrignal, str(getCalssName(indexVal)), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 0, 0), 8, cv2.LINE_AA)

    cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2,
                cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    cv2.waitKey(1)

