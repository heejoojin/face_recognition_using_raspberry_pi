import cv2 as cv
import numpy as np
from PIL import Image
import os

path = 'Data'

recognizer = cv.face.LBPHFaceRecognizer_create()

face_cascade = cv.CascadeClassifier('/home/pi/fdcam/opencv/data/haarcascades/'
                                    + 'haarcascade_frontalface_default.xml')

def get_image_and_label(path):
    
    images = [os.path.join(path, f) for f in os.listdir(path)]
    
    faces = []
    ids = []
    
    count = 0
    
    for image in images:
        
        image_PIL = Image.open(image).convert('L') # grayscale
        # PIL = python image library
        
        image_np = np.array(image_PIL,'uint8')
        # print(image_np)
        # RGB (color) images become 3D ndarray (row (height) x column (width) x color (3)).
        # Black and white (grayscale) images become 2D ndarray
        # (row (height) x column (width)).
        
        id = int(os.path.split(image)[-1].split("_")[1])
        faces_detected = face_cascade.detectMultiScale(image_np)
        
        for (x,y,w,h) in faces_detected:
            faces.append(image_np[y:y+h, x:x+w])
            ids.append(id)
        
    return faces, ids

print ("\n Training ...")

faces, ids = get_image_and_label(path)
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.write('Train/train.yml')

# Print the numer of faces trained and End program
print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))