from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import sys
import imutils
import numpy as np



cascPath = "haarcascade_frontalface_default.xml"
font = cv2.FONT_HERSHEY_SIMPLEX

faceCascade = cv2.CascadeClassifier(cascPath)

casc_smile_path = "haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(casc_smile_path)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (260, 220)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(260, 220))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    image = frame.array
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 100), 1)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        smile = smile_cascade .detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )   
       
       
        for (x2, y2, w2, h2) in smile:
            
            cv2.rectangle(roi_color, (x2,y2), (x2+w2, y2+h2), (0, 255, 100), 1)
            time.sleep(0.02)
                          
       

       
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    
    

    if key == ord("q"):
        break
        
  
        

