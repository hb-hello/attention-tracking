from .gaze_tracking import GazeTracking
import cv2
import logging
import os
import time
from scipy.spatial import distance

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
FRAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated_frames')


def detect(gray, frame):
    all_roi_faces=[]
    all_coordinates=[]
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i,(x, y, w, h) in enumerate(faces):
        roi_color = frame[y:y+h, x:x+w]
        all_roi_faces.append(roi_color)
        all_coordinates.append([x+(w/2), y+(h/2)])
    return all_roi_faces,all_coordinates

def getGazeAttention(image, frame_counter):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts frames to gray scale
    image_x = gray.shape[0]  
    image_y = gray.shape[1]

    x_mid = image_x/2 
    y_mid = image_y/2

    del_x=0
    del_y=0

    threshold = 0.16

    gaze = GazeTracking()
    logging.debug("Gaze Tracking")

    gaze.refresh(image)
    text=""

    t1=0
    t2=0
    R=0
    C=0
    L=0

    # attention_percent = ((len(all_faces) - total_del)*100)/(len(all_faces)+0.1)
    logging.debug('Pupils?' + str(gaze.pupils_located))
    text = 'Could not detect pupils.'

    if(gaze.pupils_located):
        text = f'{gaze.eye_left.center[0]}, {gaze.eye_left.center[1]} : {gaze.eye_left.pupil.x}, {gaze.eye_left.pupil.y}'
        pupil_left = [gaze.eye_left.pupil.x, gaze.eye_left.pupil.y]
        pupil_right = [gaze.eye_right.pupil.x, gaze.eye_right.pupil.y]
        metric_left = distance.euclidean(gaze.eye_left.center, pupil_left) / (((gaze.eye_left.center[0] ** 2)+(gaze.eye_left.center[0] ** 2)) ** 0.5)
        metric_right = distance.euclidean(gaze.eye_right.center, pupil_right) / (((gaze.eye_right.center[0] ** 2)+(gaze.eye_right.center[0] ** 2)) ** 0.5)
        metric = (metric_left + metric_right) / 2
        if metric < threshold:
            attention_percent = 1.0 - metric
        else:
            attention_percent = max(0.6 - metric, 0.0)
    else:
        attention_percent = 0.0

    attention_percent = attention_percent * 100

    formatted_time = time.strftime('%b%d-%H-%M', time.localtime(time.time()))
    image_file = os.path.join(FRAMES_PATH, f'frame_{formatted_time}_{frame_counter}.png')
    annotated_frame = cv2.putText(gaze.annotated_frame(), f'Attention: {str(attention_percent)}', (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    annotated_frame = cv2.putText(annotated_frame, f'{text}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    # annotated_frame = cv2.putText(annotated_frame, f'{metric_left}, {metric_right}', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    cv2.imwrite(image_file, annotated_frame)

    logging.debug(f"GAZE ATTENTION: {attention_percent}")
    return attention_percent