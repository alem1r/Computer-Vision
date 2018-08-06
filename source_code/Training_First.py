from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import sys
import imutils
import numpy as np
import time
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


camera = PiCamera()
camera.resolution = (260, 220)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(260, 220))

# allow the camera to warmup
time.sleep(0.1)
count = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    image = frame.array
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3,5)

    for (x, y, w, h) in faces:
        cropped_face = image[y:y+h, x:x+w]
        
        count+=1
        face = cv2.resize(image, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = '/home/pi/Desktop/test/images/'    #path where data are stored
        name =    str(count)+'.jpg'
        cv2.imwrite(os.path.join(file_name_path,name), face)
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper', face)

	
    key = cv2.waitKey(1) & 0xFF
    
	# clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    
    
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
