## "Face Detector" - Haar Cascade using Python

What if the machine is able to identify and recognize your face automatically in an image without human involvement? Well, this is exactly what is being implemented in this project.

### :one: Introduction
The "Face Detector" is an OpenCV project that identifies human faces from an image using the cascade classifier in OpenCV Python. Face detection has much significance in different fields of today’s world. It is a significant step in several applications, face recognition (also used as biometrics), photography (for auto-focus on the face), face analysis (age, gender, emotion recognition), video surveillance, etc.

### :two: Prerequisites
Python, OpenCV, Image Classification and Deep learning knowledge

### :three: What is Haar cascade?
The Haar Cascade classifier is an effective tool for detecting objects. Haarcascade file can be download from [here](haarcascade_frontalface_default.xml). The method was introduced by Paul Viola and Michael Jones in their paper Rapid Object Detection using a Boosted Cascade of Simple Features. An Haar Cascade is a machine learning approach that uses a large set of positives and negatives to train a classifier. The positive images contain the images we want our classifier to identify and the negative images contain everything else, which does not contain the object we want to identify.

### :four: Four stages of Haar Cascade
- Haar-feature selection: A Haar-like feature consists of dark regions and light regions. It produces a single value by taking the difference of the sum of the intensities of the dark regions and the sum of the intensities of light regions. It is done to extract useful elements necessary for identifying an object.
- Creation of Integral Images: A given pixel in the integral image is the sum of all the pixels on the left and all the pixels above it. Since the process of extracting Haar-like features involves calculating the difference of dark and light rectangular regions, the introduction of Integral Images reduces the time needed to complete this task significantly.
- AdaBoost Training: This algorithm selects the best features from all features. It combines multiple “weak classifiers” (best features) into one “strong classifier”. The generated “strong classifier” is basically the linear combination of all “weak classifiers”.
- Cascade Classifier: It is a method for combining increasingly more complex classifiers like AdaBoost in a cascade which allows negative input (non-face) to be quickly discarded while spending more computation on promising or positive face-like regions. It significantly reduces the computation time and makes the process more efficient

### :five: Conclusion
In this project, OpenCV module is used to detect faces. As OpenCV already contains many pre-trained classifiers for face, eyes, smile, etc., it becomes very eay to use and implement. Thank you!
