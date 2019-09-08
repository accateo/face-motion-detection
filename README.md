# Face and Motion Detection in Python
Script to run a server that make face and motion detection in Python

## Requirements:

Tested with Python version 3.7.3. I also used pip to install modules.

- OpenCV installed

  *apt-get install python3-opencv*
- Flask (webserver) installed 

  *pip3 install flask*
- Setuptools installed

  *pip3 install setuptools*
- Apscheduler installed

  *pip3 install apscheduler*
- Imutils installed

  *pip3 install imutils*
- Pandas installed

  *pip3 install pandas*

...and a webcam

## Content

- webServer.py
  
  Main file. Manage the web server and contains the API to start and stop detection

- detection.py
  
  File that contains all the code to face and motion detection
  
- haarcascade_frontalface_default.xml and haarcascade_profileface.xml

  Two xml descriptive files which are used to detect front and side faces. 
  
  
 ## How to run
 
 *python3 webServer.py*
 
 To start detection:
 
 http call like this
 
 http://127.0.0.1:5000/startDetector
 
 To stop detection:
 
 http://127.0.0.1:5000/stopDetector
 
 To get info about detection (number of faces, motion detected, distance...):
 
 http://127.0.0.1:5000/getInfo
 
 127.0.0.1 for local or the IP where the server runs...
 
