import os
import cv2 as cv
import numpy as np
people = ['Narendra Modi', 'Nirmala Sitharaman', 'Ram Nath Kovind', 'Smriti Irani']
DIR = r'D:\Facerecognition\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []  # image faces
labels = []   # who's face

def create_train():
    for person in people:
        path = os.path.join(DIR, person)  # grabbing each img from folder
        label = people.index(person)   # label is the index

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

            for (x, y, w, h) in faces_rect:
                face_region = gray[y:y+h, x:x+w]
                features.append(face_region)
                labels.append(label)

create_train()
print('Training is done!!!')

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
# train face recognizer to labels and features
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)




