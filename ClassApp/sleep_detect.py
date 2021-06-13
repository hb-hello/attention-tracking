from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import logging

flag = 0
yawn_flag = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2],mouth[10])
    C = distance.euclidean(mouth[4],mouth[8])
    B = distance.euclidean(mouth[3],mouth[9])
    mar = (A+B+C)/3
    return mar


def getSleepNumber(frames):
    thresh = 0.25
    n_sleep=0
    frame_check = 2
    count_mouth = 0
    total_yawn = 0
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("ClassApp/models/shape_predictor_68_face_landmarks.dat")  # This file is the crux of the code

    sleepy_coordinates = []
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    
    global flag
    global yawn_flag

    for frame in frames:
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            coordinates=[leftEye, rightEye]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouth = shape[mStart: mEnd]
            mouthEAR = mouth_aspect_ratio(mouth)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    n_sleep += 1
                    flag = 0
                    sleepy_coordinates.append(coordinates)
            else:
                flag = 0


            logging.debug('Sleep flag?' + str(flag)) 

            if mouthEAR > 30:
                count_mouth += 1
                if count_mouth >= 10:
                    if yawn_flag < 0:
                        yawn_flag = 1
                        total_yawn += 1
                    else:
                        yawn_flag = 1
                else:
                    yawn_flag = -1
            else:
                count_mouth = 0
                yawn_flag = -1

    logging.debug('Sleep?' + str(n_sleep))

    return n_sleep, sleepy_coordinates
