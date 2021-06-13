from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse
from concurrent.futures import ThreadPoolExecutor
import os
import time
import threading

from ClassApp.ml_algo import MakeAttention

from PIL import Image
import base64
import numpy as np
import json
import time
import ast
import cv2

FRAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames')

@csrf_exempt
def capture_list(request):
    t0 = time.time()
    np_frames = []
    base_64_image = request.POST.get('list').split(',')[1]
    frame_counter = int(request.POST.get('counter'))
    t1 = time.time()
    np_frame = np.frombuffer(base64.b64decode(base_64_image), np.uint8)
    print("Received frame")
    # width = int(request.POST.get('width'))
    # height = int(request.POST.get('height'))
    

    # temp1 = [frame[i:i + 3] for i in range(0, len(frame), 4)]
    # temp2 = [temp1[i:i + width] for i in range(0, len(temp1), width)]
    # np_frame = Image.fromarray(np.array(temp2, dtype=np.uint8))
    # np_frame = np.array(temp2, dtype=np.uint8)
    np_frames.append(cv2.imdecode(np_frame, cv2.IMREAD_COLOR))
        # Image.fromarray(np.array(temp2, dtype=np.uint8)).save(os.path.join(FRAMES_PATH, f'{time.time()}.png'))
    # print("Sending frames to ML Model", len(np_frames))
    t2 = time.time()
    attention_thread = threading.Thread(
            target=MakeAttention, kwargs=dict(frames=np_frames, frame_counter=frame_counter))
    attention_thread.start()
    # MakeAttention(np_frames)
    t3 = time.time()

    print(t1-t0, t2-t1, t3-t2)
    return HttpResponse('OK')


@csrf_exempt
def capture_attendance(request):
    np_frames = []
    list_frames = ast.literal_eval(request.POST.get('list'))
    print("Received", len(list_frames), "frames")
    width = int(request.POST.get('width'))
    height = int(request.POST.get('height'))

    for frame in list_frames:
        img = list(frame['data'].values())
        temp1 = [img[i:i + 3] for i in range(0, len(img), 4)]
        temp2 = [temp1[i:i + width] for i in range(0, len(temp1), width)]
        # np_frame = Image.fromarray(np.array(temp2, dtype=np.uint8))
        np_frame = np.array(temp2, dtype=np.uint8)
        np_frames.append(np_frame)
        Image.fromarray(np.array(temp2, dtype=np.uint8)).save("frames/Attendance {0}.png".format(time.time()))
    # MakeAttention(np_frames)
    StartAttendance(np_frames[0])

    return HttpResponse('OK')

# @csrf_exempt
# def capture_face(request):
#     np_frames = []
#     list_frames = ast.literal_eval(request.POST.get('list'))
#     print("Received", len(list_frames), "frames")
#     width = int(request.POST.get('width'))
#     height = int(request.POST.get('height'))
#     course_time = request.POST.get('time')
#     course = request.POST.get("course")
#
#     for frame in list_frames:
#         img = list(frame['data'].values())
#         temp1 = [img[i:i + 3] for i in range(0, len(img), 4)]
#         temp2 = [temp1[i:i + width] for i in range(0, len(temp1), width)]
#         # np_frame = Image.fromarray(np.array(temp2, dtype=np.uint8))
#         np_frame = np.array(temp2, dtype=np.uint8)
#         np_frames.append(np_frame)
#         Image.fromarray(np.array(temp2, dtype=np.uint8)).save("Attendance {0}.png".format(time.time()))
#     # MakeAttention(np_frames)
#     DetectAttendance(np_frames, course, course_time)
#
#     return HttpResponse('OK')
