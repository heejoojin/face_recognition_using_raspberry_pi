import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Train/train.yml')
face_cascade = cv2.CascadeClassifier('/home/pi/fdcam/opencv/data/haarcascades/'
                                    + 'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

# Index = ID
names = ['None', 'Heejoo', 'David', 'Munhee', 'Gil']

cam = cv2.VideoCapture(0)
cam.set(3, 640) # Set width
cam.set(4, 480) # Set height

# Define mininum window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    _, img = cam.read()
    img = cv2.flip(img, 1) # Flip horizontally
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        radius = int(round((w + h) * 0.25))
        frame = cv2.circle(img, center, radius, (255, 255, 255), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Check confidence
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(confidence), (x + w - 5, y + h - 5), font, 1, (255,255,255), 2)  
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x + 5,y - 5), font, 1, (255,255,255), 2)  
    
    cv2.imshow('Face Recognition',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
print("\n Exiting")
cam.release()
cv2.destroyAllWindows()