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
    refine_landmarks=True,  # Enables detailed eyes points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Funzione per calcolare la distanza euclidea tra due punti
def distanza_euclidea(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 3 - Open the video source
cap = cv2.VideoCapture(0)  # Local webcam (index start from 0)

# Definizione punti di interesse occhi/bocca/naso
punti_YawPitch = {33:None,263:None,1:None,61:None,291:None,199:None}

# Punti per il calcolo dell'EAR
punti_EAR_DX = {160:None,158:None,153:None,163:None,33:None,133:None}
punti_EAR_SX = {263:None,384:None,387:None,373:None,381:None,362:None}

# Variabili per calcolare PERCLOS
i1, i2, i3, i4 = 0, 0, 0, 0
t1, t2, t3, t4 = 0, 0, 0, 0
eye_state = -1
perc_time = 0
threshold_80 = 0.18 # Soglie più o meno a caso
threshold_20 = 0.13


# 4 - Iterate (within an infinite loop)
while cap.isOpened():
    # 4.1 - Get the new frame
    success, image = cap.read() 
    
    start = time.time()

    if image is None:
        break

    # To improve performance
    image.flags.writeable = False
    
    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    img_h, img_w, img_c = image.shape

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
            time.sleep(1/25)  # [s]

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime if totalTime > 0 else 0

        # Calcolo EAR con distanza euclidea
        EAR_DX = (
            distanza_euclidea(punti_EAR_DX[160], punti_EAR_DX[163]) +
            distanza_euclidea(punti_EAR_DX[158], punti_EAR_DX[153])
        ) / (
            2 * distanza_euclidea(punti_EAR_DX[33], punti_EAR_DX[133])
        )
        EAR_SX = (
            distanza_euclidea(punti_EAR_SX[384], punti_EAR_SX[381]) +
            distanza_euclidea(punti_EAR_SX[387], punti_EAR_SX[373])
        ) / (
            2 * distanza_euclidea(punti_EAR_SX[362], punti_EAR_SX[263])
        )

        cv2.putText(image, f'EAR DX : {float(EAR_DX):.3f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(image, f'EAR SX : {float(EAR_SX):.3f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # Calcolo PERCLOS
        if EAR_DX >= threshold_80 and EAR_SX >= threshold_80:
            if eye_state == 4 or eye_state == -1:
                print(f"eye state, from {str(eye_state)} to 1")
                eye_state = 1
                i1 = time.time()
                print("1: "+str(t1))
                if i1 > 0 and i4 > 0:
                    t4 = i4 - i2
        elif (EAR_DX < threshold_80 and EAR_DX > threshold_20) and (EAR_SX < threshold_80 and EAR_SX > threshold_20):
            if eye_state == 1:
                print(f"eye state, from {str(eye_state)} to 2")
                eye_state = 2
                i2 = time.time()
                print("i2: "+str(i2))
                if i1 > 0 and i2 > 0:
                    t1 = i2 - i1
            elif eye_state == 3:
                print(f"eye state, from {str(eye_state)} to 4")
                eye_state = 4
                i4 = time.time()
                print("i4: "+str(i4))
                if i3 > 0 and i4 > 0:
                    t3 = i4 - i3
        else: #elif EAR_DX <= threshold_20 and EAR_SX <= threshold_20:
            if eye_state == 2:
                print(f"eye state, from {str(eye_state)} to 3")
                eye_state = 3
                i3 = time.time()
                print("i3: "+str(i3))
                if i3 > 0 and i2 > 0:
                    t2 = i3 - i2

        if t1 > 0 and t2 > 0 and t3 > 0 and t4 > 0:
            perc_time = (t3 - t2) / (t4 - t1)
            cv2.putText(image, f'PERCLOS : {perc_time:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        if t1 > 2000:
            cv2.putText(image, f'SBATTI LE PALPEBRE SONO PREOCCUPATO', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Show FPS
        cv2.putText(image, f'FPS : {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 4.5 - Show the frame to the user
        cv2.imshow('Driver Monitoring - PERCLOS Calculation', image)       
                    
    if cv2.waitKey(5) & 0xFF == 27:  # Exit if ESC is pressed
        break

# 5 - Close properly source and eventual log file
cap.release()
