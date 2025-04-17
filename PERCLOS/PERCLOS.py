import cv2
import mediapipe as mp
import numpy as np
import time
import statistics as st
import os

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

cap = cv2.VideoCapture(0)  # Local webcam (index start from 0)

# Punti di interesse per gli occhi, bocca e naso
punti_YawPitch = [33, 263, 1, 61, 291, 199]

# Punti per il calcolo dell'EAR
punti_EAR_DX = {160: None, 158: None, 153: None, 163: None, 33: None, 133: None}
punti_EAR_SX = {263: None, 384: None, 387: None, 373: None, 381: None, 362: None}

# Variabili per calcolare PERCLOS
i1, i2, i3, i4 = 0, 0, 0, 0
t1, t2, t3, t4 = 0, 0, 0, 0
eye_state = -1
perc_time = 0
threshold_80 = 0.18  # Soglia di rilevamento occhi chiusi
threshold_20 = 0.13  # Soglia di attenzione media

scale_v = 1 # Fattore di scala per la visualizzazione dei vettori

distraction_threshold = 30  # in degrees
scale_factor = 0.7  # Fattore di scala per la visualizzazione dei vettori

# Head pose calculation variables
selected_landmarks = [1, 33, 263, 4]  # Aggiunto l'indice 4 per un punto aggiuntivo

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    if image is None:
        break

    image.flags.writeable = False

    results = face_mesh.process(image)

    image.flags.writeable = True

    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []
    right_eye_2d = []
    right_eye_3d = []
    left_eye_2d = []
    left_eye_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            time.sleep(0.05)  # Delay to simulate processing time
            # Calcolare EAR
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in punti_EAR_DX.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    punti_EAR_DX[idx] = int(lm.x * img_w), int(lm.y * img_h)
                if idx in punti_EAR_SX.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    punti_EAR_SX[idx] = int(lm.x * img_w), int(lm.y * img_h)
                # _______________________________________________________________________________
                # Raccolta punti orientamento testa
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                # Raccolta punti occhi DX/SX
                if idx == 468 or idx == 33 or idx == 145 or idx == 133 or idx == 159:
                    if idx == 468:
                        right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d.append([x, y])
                    right_eye_3d.append([x, y, lm.z])
                if idx == 473 or idx == 362 or idx == 374 or idx == 263 or idx == 386:
                    if idx == 473:
                        left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                        left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d.append([x, y])
                    left_eye_3d.append([x, y, lm.z])
                # _______________________________________________________________________________
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
            left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
            right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
            right_eye_3d = np.array(right_eye_3d, dtype=np.float64)
            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)# The distorsion parameters
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
            success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
            rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye,Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
            angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)
            pitch = angles[0] * 1800
            yaw = -angles[1] * 1800
            roll = 180 + (np.arctan2(punti_EAR_DX[33][1] - punti_EAR_SX[263][1], punti_EAR_DX[33][0] - punti_EAR_SX[263][0]) * 180 / np.pi)
            if roll > 180:
                roll = roll - 360
            pitch_left_eye = angles_left_eye[0] * 1800
            yaw_left_eye = angles_left_eye[1] * 1800
            pitch_right_eye = angles_right_eye[0] * 1800
            yaq_right_eye = angles_right_eye[1] * 1800
            # DISPLAY DIRECTIONS
            # Nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + yaw * scale_factor), int(nose_2d[1] - pitch * scale_factor))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Left eye direction
            left_pupil_3d_projection, jacobian_left_eye = cv2.projectPoints(left_pupil_3d, rot_vec_left_eye, trans_vec_left_eye, cam_matrix, dist_matrix)
            p1_left_eye = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
            p2_left_eye = (int(left_pupil_2d[0] + yaw_left_eye * scale_factor), int(left_pupil_2d[1] - pitch_left_eye * scale_factor))
            cv2.line(image, p1_left_eye, p2_left_eye, (0, 0, 255), 3)

            # Right eye direction
            right_pupil_3d_projection, jacobian_right_eye = cv2.projectPoints(right_pupil_3d, rot_vec_right_eye, trans_vec_right_eye, cam_matrix, dist_matrix)
            p1_right_eye = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
            p2_right_eye = (int(right_pupil_2d[0] + yaq_right_eye * scale_factor), int(right_pupil_2d[1] - pitch_right_eye * scale_factor))
            cv2.line(image, p1_right_eye, p2_right_eye, (0, 255, 0), 3)

            if abs(pitch / 10) > distraction_threshold or abs(yaw / 10) > distraction_threshold or abs(roll) > distraction_threshold:
                cv2.putText(image, 'ATTENTO! SEI DISTRATTO!', (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)


            # Calcolare EAR (distanza euclidea tra gli occhi)
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

            # Calcolare PERCLOS
            if EAR_DX >= threshold_80 and EAR_SX >= threshold_80:
                if eye_state == 4 or eye_state == -1:
                    print(f"eye state 1")
                    eye_state = 1
                    i1 = time.time()
                    if i1 > 0 and i4 > 0:
                        t4 = i1 - i4
                        print("t4: "+str(t4))

            elif (EAR_DX < threshold_80 and EAR_DX > threshold_20) and (EAR_SX < threshold_80 and EAR_SX > threshold_20):
                if eye_state == 1:
                    print(f"eye state 2")
                    eye_state = 2
                    i2 = time.time()
                    if i1 > 0 and i2 > 0:
                        t1 = i2 - i1
                        print("t1: "+str(t1))
                elif eye_state == 3:
                    print(f"eye state 4")
                    eye_state = 4
                    i4 = time.time()
                    if i3 > 0 and i4 > 0:
                        t3 = i4 - i3
                        print("t3: "+str(t3))
            else:
                if eye_state == 2:
                    print(f"eye state 3")
                    eye_state = 3
                    i3 = time.time()
                    if i3 > 0 and i2 > 0:
                        t2 = i3 - i2
                        print("t2: "+str(t2))

            if t1 > 0 and t2 > 0 and t3 > 0 and t4 > 0:
                if t4 > t1 and t3 > t2:
                    perc_time = (t3 - t2) / (t4 - t1)
                else:
                    perc_time = 0  # Default value or handle the case appropriately
                cv2.putText(image, f'PERCLOS : {perc_time:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            if t1 > 10:  # In secondi
                cv2.putText(image, f'SBATTI LE PALPEBRE SONO PREOCCUPATO', (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime if totalTime > 0 else 0

    cv2.putText(image, f'FPS : {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow('Driver Monitoring - PERCLOS Calculation', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit if ESC is pressed
        break

cap.release()
cv2.destroyAllWindows()
