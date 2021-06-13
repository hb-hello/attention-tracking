from .gaze_tracking import GazeTracking
import cv2
import logging
import os

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# face_cascade = cv2.CascadeClassifier(os.path.join(MODELS_PATH, 'haarcascade_frontalface_default.xml'))

def detect(gray, frame):
    all_roi_faces=[]
    all_coordinates=[]
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i,(x, y, w, h) in enumerate(faces):
        roi_color = frame[y:y+h, x:x+w]
        all_roi_faces.append(roi_color)
        all_coordinates.append([x+(w/2), y+(h/2)])
    return all_roi_faces,all_coordinates

def getGazeAttention(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts frames to gray scale
    image_x = gray.shape[0]  
    image_y = gray.shape[1]

    x_mid = image_x/2 
    y_mid = image_y/2

    del_x=0
    del_y=0

    # all_faces, all_coordinates = detect(gray, image)   #call function
    gaze = GazeTracking()
    logging.debug("Gaze Tracking")
    
    # for i,frame in enumerate(all_faces):
    # gaze.refresh(frame)
    gaze.refresh(image)
    text=""

    t1=0
    t2=0
    R=0
    C=0
    L=0

    ## Code for attention as per person position
    # if all_coordinates[i][0]<x_mid:
    #     t1 = abs(all_coordinates[i][0]-x_mid)/x_mid
    #     t2 = abs(all_coordinates[i][1])/y_mid
    #     R=0
    #     C=0.5
    #     L=1

    # else:
    #     t1 = abs(all_coordinates[i][0] - x_mid) / x_mid
    #     t2 = abs(all_coordinates[i][1]) / y_mid
    #     R = 1
    #     C = 0.5
    #     L = 0

    # if gaze.is_right():
    #     del_x = del_x + t1*R
    #     del_y = del_y + t2*R
    # elif gaze.is_left():
    #     del_x = del_x + t1*L
    #     del_y = del_y + t2*L
    # elif gaze.is_center():
    #     del_x = del_x + t1*C
    #     del_y = del_y + t2*C

    # total_del = (del_x + del_y)/2
   
    # attention_percent = ((len(all_faces) - total_del)*100)/(len(all_faces)+0.1)
    logging.debug('Pupils?' + str(gaze.pupils_located))
    logging.debug(gaze.horizontal_ratio)

    if(gaze.pupils_located):
        attention_percent = 100 * (1.0 - (abs(gaze.horizontal_ratio() - 0.5) + abs(gaze.vertical_ratio() - 0.5)))
    else:
        attention_percent = 0.0

    logging.debug(f"GAZE ATTENTION: {attention_percent}")
    return attention_percent