import cv2
from tensorflow.keras.preprocessing.image import load_img , img_to_array
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = load_img('test.jpg')
img_arr = img_to_array(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 6)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.plot(img_arr[y:y+h,x:x+w ,:])
    plt.show()

cv2.imshow('img', img)
cv2.waitKey()