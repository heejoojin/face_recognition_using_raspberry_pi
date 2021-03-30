import cv2 as cv

# Normally one camera will be connected (as in my case).
# So I simply pass 0 (or -1). you can select the second camera by passing 1 and so on.
cap = cv.VideoCapture(0)

cap.set(3,640) # Set Width
cap.set(4,480) # Set Height

face_cascade = cv.CascadeClassifier('/home/pi/fdcam/opencv/data/haarcascades/'
                                    + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('/home/pi/fdcam/opencv/data/haarcascades/'
                                    + 'haarcascade_eye_tree_eyeglasses.xml')
# smiles_cascade = cv.CascadeClassifier('/home/pi/fdcam/opencv/data/haarcascades/'
                                      # + 'haarcascade_smile.xml')
                                      
# For each person, enter one numeric face id
face_id = input('\n Enter user ID and Press <return> ->  ')
print('\n Initializing face capture. Look at the camera. ')

count = 0

while (True):
    # This code initiates an infinite loop
    # (to be broken later by a break statement),
    # where we have ret and frame being defined as the cap.read().
    
    # ret (=_) is a boolean regarding whether or not there was a return at all,
    # at the frame is each frame that is returned.
    # If there is no frame, you wont get an error, you will get False, None.
    _, frame = cap.read()
    
    frame = cv.flip(frame, 1) # Flip horizontally for my own convenience.

    # We define a new variable, gray, as the frame, converted to gray.
    # Notice this says BGR2GRAY.
    # It is important to note that OpenCV reads colors as BGR (Blue Green Red),
    # where most computer applications read as RGB (Red Green Blue).
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # cv.equalizeHist() makes the image clearer.
    # frame_gray = cv.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    # Default
    # image, cascade, storage, scale_factor=1.1, min_neighbors=3, flags=0, min_size=(0, 0)
    
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        # (image, center_coordinates, axesLength,
        # angle, startAngle, endAngle, color, thickness)
        # (255, 0, 255) = color code
        
        face_roi = frame_gray[y:y+h, x:x+w]
        color_roi = frame[y:y+h, x:x+w]
        # ROI = region of interest
        
        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(face_roi)
        
        # smiles = smiles_cascade.detectMultiScale(face_roi, scaleFactor = 1.5, minNeighbors = 20)
        
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 2)
        
        # for (x3, y3, w3, h3) in smiles:vid
            # cv.rectangle(color_roi, (x3, y3), (x3 + w3, y3 + h3), (255, 255, 255), 2) 
        
        count += 1
        # cv2.resize(img, (150, 150), interpolation=cv2.INTER_LINEAR)
        cv.imwrite("Data/user_" + str(face_id) + '_'
                + str(count) + ".jpg", frame_gray[y:y+h,x:x+w])
    cv.imshow('Face Detection', frame)
    
    key = cv.waitKey(30) & 0xff
    if key == 27: # Press 'ESC' to quit
        break
    
    elif count >= 30: # Take 30 face samples and stop the camera
         break
    
print('\n Exiting Program')
cap.release()
cv.destroyAllWindows()