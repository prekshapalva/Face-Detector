import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Narendra Modi', 'Nirmala Sitharaman', 'Ram Nath Kovind', 'Smriti Irani']

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img = cv.imread(r'D:\Facerecognition\val\Nirmala Sitharaman\NS (17).jpg')
img = cv.imread(r'D:\Facerecognition\val\Ram Nath Kovind\RNK (18).png')
# img = cv.imread(r'D:\Facerecognition\val\Ram Nath Kovind\RNK (18).png')
# img = cv.imread(r'D:\Facerecognition\val\Narendra Modi\NM (17).png')
# img = cv.imread(r'D:\Facerecognition\val\Narendra Modi\NM (16).png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces on image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 10)

for (x, y, w, h) in faces_rect:
    face_region = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(face_region)
    print(f'label = {people[label]} with confidence of {confidence}')
    cv.putText(img, str(people[label]), (25, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 0, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))

cv.imshow('Detected Face', img)

cv.waitKey(0)
