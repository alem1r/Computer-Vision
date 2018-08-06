import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #this file must be in the same directory of the source code
model = cv2.createLBPHFaceRecognizer()
model.load('/home/pi/Desktop/test/trainer.yml')  #specify your path
camera = PiCamera()
camera.resolution = (240, 220)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(240, 220))

def face_detector(img, size=0.5):

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)
	if faces is ():
		return img,[]

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
		roi = img[y:y+h, x:x+w]
		roi = cv2.resize(roi, (200,200))
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	return img,roi



for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	image = frame.array
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray, 1.2,5)
	for (x, y, w, h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
	
                results = model.predict(gray)
                print(results)
                if results[1] < 500:
                        confidence = int(100 * (1-(results[1])/300))
                        display_string = str(confidence) + '% Confidence'


                if confidence > 60:

                        cv2.putText(image, "my creator", (70,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255) , 2)
                        
                        cv2.imshow('Face Cropper',image)
                else:
                        cv2.putText(image, "Locked", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, 	(0,0,255) , 2)
                        cv2.imshow('Face Cropper',image)
	
        key = cv2.waitKey(1) & 0xFF

       
        rawCapture.truncate(0)




        if key == ord("q"):
                break
    

