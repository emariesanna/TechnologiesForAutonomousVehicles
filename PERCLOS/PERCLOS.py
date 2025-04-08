#**************************************************************************************
#
#   T4AV - Driver Monitoring Systems using AI (code sample)
#
#   File: PERCLOS.py
#   Author: Jacopo Sini, Emanuele Sanna, Marco Silveri
#   Company: Politecnico di Torino
#   Date: 8/5/2025
#
#**************************************************************************************

# 1 - Import the needed libraries
import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os

# 2 - Set the desired setting
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enables  detailed eyes points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the list of available capture devices (comment out)
#index = 0
#arr = []
#while True:
#    dev = cv2.VideoCapture(index)
#    try:
#        arr.append(dev.getBackendName)
#    except:
#        break
#    dev.release()
#    index += 1
#print(arr)

# Definizione punti di interesse occhi/bocca/naso
# 33: angolo occhio DX
# 263: angolo occhio SX
# 1: punta naso
# 61: angolo bocca DX
# 291: angolo bocca SX
# 199: mento
punti_YawPitch = {33:None,263:None,1:None,61:None,291:None,199:None}

punti_EAR_DX = {160:None,158:None,153:None,163:None,33:None,133:None}
punti_EAR_SX = {263:None,384:None,387:None,373:None,380:None,362:None}

# 3 - Open the video source
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)

# 4 - Iterate (within an infinite loop)
while cap.isOpened(): 
    
    # 4.1 - Get the new frame
    success, image = cap.read() 
    
    start = time.time()

    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    #else: #needed with some cameras/video input format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    img_h, img_w, img_c = image.shape

    point_RER = [] # Right Eye Right
    point_REB = [] # Right Eye Bottom
    point_REL = [] # Right Eye Left
    point_RET = [] # Right Eye Top

    point_LER = [] # Left Eye Right
    point_LEB = [] # Left Eye Bottom
    point_LEL = [] # Left Eye Left
    point_LET = [] # Left Eye Top

    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center


    # 4.3 - Get the landmark coordinates
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                if idx in punti_YawPitch.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=5, color=(0, 0, 255), thickness=-1)
                if idx in punti_EAR_DX.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                if idx in punti_EAR_SX.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)

                if idx in punti_YawPitch.keys():
                    punti_YawPitch[idx] = int(lm.x * img_w), int(lm.y * img_h)
                if idx in punti_EAR_DX.keys():
                    punti_EAR_DX[idx] = int(lm.x * img_w), int(lm.y * img_h)
                if idx in punti_EAR_SX.keys():
                    punti_EAR_SX[idx] = int(lm.x * img_w), int(lm.y * img_h)

            # Speed reduction (comment out for full speed)
            time.sleep(1/25) # [s]

        end = time.time()
        totalTime = end-start

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0
        
        
        #print("FPS:", fps)

        cv2.putText(image, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI code sample', image)       
                    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly source and eventual log file
cap.release()
#log_file.close()
    
# [EOF]
