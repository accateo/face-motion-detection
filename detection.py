import apscheduler
import cv2
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np
import imutils
from datetime import datetime
from pandas import pandas as pd

motion = 0
data = {}
counterFrame = 0
thresholdFrameBackground = 300
thresholdContourArea = 1500
enable = 0
# data returned
# data = {"N_faces":0,"dim":[213], "motion": 1, "array_center": [0,0], "array_coordinates": [0,0], "positions_faces": ["top left"] ,"array_center_motion": [0], "direction" : "left"}
# N_faces - number of faces
# dim - dimensions of squares around the faces
# motion - 0 not motion 1 motion
# array_center - coordinates of the center of the square around the face detected
# positions_faces - positions of the faces "top left", "top right"...
# array_center_motion - coordinates of the center of the motion detected
# direction - direction of motion detection "left", "right", "top", "bottom"
def DetectFace(faceCascade, faceProfileCascade, minW,minH):
    global motion
    global data
    global counterFrame
    global thresholdFrameBackground
    global thresholdContourArea
    global static_back
    global direction
    global positions

    ret, img = cam.read()
    #faces positions in the space
    positions = []
    face_found = False
    #face detection
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    w_ = []
    array_center_motion = []
    array_center_faces = []
    array_coordinates_faces = []
    for(x,y,w,h) in faces:
        pair_data = []
	#draw rectangle around the faces
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        pair_data = (((x+x+w)/2),((y+y+w)/2))
        pair_coordinates = (x,y)
        array_coordinates_faces.append(pair_coordinates)
        array_center_faces.append(pair_data)
        #position of the faces in the space
        positions.append(positionFromCoordinates(pair_data))
        detect_face = img[int(y):int(y+h), int(x):int(x+w)]
        detect_face =cv2.cvtColor(detect_face, cv2.COLOR_RGB2BGR)
        detect_face = cv2.resize(detect_face, (224,224))
        w_.append(int(w))
        face_found = True
    print(data)
    #profile faces
    if face_found == False:
	print("profile face detected")
        faces = faceProfileCascade.detectMultiScale(
            img,
            scaleFactor = 1.3,
            minNeighbors = 4,
            flags = (cv2.CASCADE_DO_CANNY_PRUNING + cv2.CASCADE_FIND_BIGGEST_OBJECT + cv2.CASCADE_DO_ROUGH_SEARCH),
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            pair_data = []
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            pair_data = (((x+x+w)/2),((y+y+w)/2))
            pair_coordinates = (x,y)
            array_coordinates_faces.append(pair_coordinates)
            array_center_faces.append(pair_data)
            positions.append(positionFromCoordinates(pair_data))
            detect_face = img[int(y):int(y+h), int(x):int(x+w)]
            detect_face =cv2.cvtColor(detect_face, cv2.COLOR_RGB2BGR)
            detect_face = cv2.resize(detect_face, (224,224))
            w_.append(int(w))
            face_found=True
    if face_found==False:
        cv2.flip(img,1,img)
	print("frontal face detected")
        faces = faceCascade.detectMultiScale(
             img,
             scaleFactor = 1.3,
             minNeighbors = 4,
             minSize = (int(minW), int(minH)),
            )
        for(x,y,w,h) in faces:
            pair_data = []
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            pair_data = (((x+x+w)/2),((y+y+w)/2))
            pair_coordinates = (x,y)
            array_coordinates_faces.append(pair_coordinates)
            array_center_faces.append(pair_data)
            positions.append(positionFromCoordinates(pair_data))
            detect_face = img[int(y):int(y+h), int(x):int(x+w)]
            detect_face =cv2.cvtColor(detect_face, cv2.COLOR_RGB2BGR)
            detect_face = cv2.resize(detect_face, (224,224))
            w_.append(int(w))
            face_found=True

    #motion detection
    #Initializing motion = 0(no motion) 
    motion = 0
    data["motion"] = 0
    direction = ""
	  
    # Converting color image to gray_scale image 
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 	  
    # In first iteration we assign the value  
    # of static_back to our first frame 
    ###
	##Find max element and index
    if len(w_):
        print("Max:{}".format(w_.index(max(w_))))
    #sometimes I update the original frame that I use to detect motion (thresholdFrameBackground)
    if static_back is None or counterFrame > thresholdFrameBackground: 
        static_back = gray	
        motion = 0
        counterFrame = 0
        data["motion"] = 0;
	#json data returned
        data = {"N_faces":len(faces),"dim":w_, "motion": motion, "array_center": array_center_faces, "array_coordinates": array_coordinates_faces, "positions_faces": positions ,"array_center_motion": array_center_motion, "direction" : ""}
        print("{} - {}".format(pd.datetime.now(), motion))
        return
    ###
	  
    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 
	  
    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
    cnts = []	  
    # Finding contour of moving object 
    cnts = cv2.findContours(thresh_frame.copy(),  
					   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = imutils.grab_contours(cnts)  
    for contour in cnts: 
	#I'm not considering very little movement
        if cv2.contourArea(contour) < thresholdContourArea: 
            continue

        motion = 1
        pair_motion_center = []
        data["motion"] = 1
        (xx, yy, ww, hh) = cv2.boundingRect(contour)
        pair_motion_center = (((xx+xx+ww)/2),((yy+yy+ww)/2))
        array_center_motion.append(pair_motion_center)
        cv2.rectangle(img, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        cv2.imwrite("./motion.jpg",img)
        print("{} - {}".format(pd.datetime.now(), motion))
        break
    #position of the motion
    if motion==1:
       if ((xx+xx+ww)/2)<200:
           direction = "left"
       else:
           if ((xx+xx+ww)/2)>300:
               direction = "right"
           else:
               direction = "center"
    data = {"N_faces":len(faces),"dim":w_, "motion": motion,"array_center": array_center_faces , "array_coordinates": array_coordinates_faces, "positions_faces": positions, "array_center_motion": array_center_motion, "direction" : direction}
    print("{} - {}".format(pd.datetime.now(), motion))
    # Increment the frame counter
    counterFrame += 1
    static_back = gray
	


def FaceDetector(scheduler, REFRESH_INTERVAL,p:int=0):
    #variabile per motion detection
    global static_back
    global counterFrame
    # Assigning our static_back to None 
    static_back = None
    #two profiles: side faces and frontal faces
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceProfileCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    millisec = 10
    counterFrame = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    StartCam(p)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    scheduler.add_job(DetectFace, 'interval', max_instances=1, seconds = REFRESH_INTERVAL, args=[faceCascade, faceProfileCascade,minW,minH], id = 'faceTimer')
    #scheduler.start()


def StartCam(p:int = 0):
    global cam
    print("p",p)
    cam = cv2.VideoCapture(p)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height



def StopCam():
    cv2.destroyAllWindows()
    cam.release()



def StopDetect(scheduler):
    StopCam()
    if len(scheduler.get_jobs()) > 0:
        scheduler.remove_job('faceTimer')
    print("Stop ---")


def getinfo():
    return data

def positionFromCoordinates(coords):
    if coords[0]>320:
       if coords[1]>210:
          return "top left"
       else:
          return "bottom left"
    else:
       if coords[1]>210:
          return "top right"
       else:
          return "bottom right"
		 
