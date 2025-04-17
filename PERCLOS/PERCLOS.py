import cv2
import mediapipe as mp
import numpy as np
import time
import statistics as st
import os

PRINT = 2 # Set to 0 to disable print statements, 1 to enable gaze tracking, 2 to enable EAR and PERCLOS tracking

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

# Computes the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_camera_matrix(img_w, img_h):
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
    [0, focal_length, img_w / 2],
    [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)# The distorsion parameters
    return cam_matrix, dist_matrix

def eyes_gaze_3d(left_eye_2d, left_eye_3d, right_eye_2d, right_eye_3d, img_w, img_h):

    left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
    left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
    right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
    right_eye_3d = np.array(right_eye_3d, dtype=np.float64)

    cam_matrix, dist_matrix = get_camera_matrix(img_w, img_h)

    success_left_eye, rot_vec_left_eye, trans_vec_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
    success_right_eye, rot_vec_right_eye, trans_vec_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)

    rmat_left_eye, jac_left_eye = cv2.Rodrigues(rot_vec_left_eye)
    rmat_right_eye, jac_right_eye = cv2.Rodrigues(rot_vec_right_eye)

    angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rmat_left_eye)
    angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rmat_right_eye)

    pitch_left_eye = angles_left_eye[0] * 1800
    yaw_left_eye = angles_left_eye[1] * 1800
    pitch_right_eye = angles_right_eye[0] * 1800
    yaw_right_eye = angles_right_eye[1] * 1800

    return pitch_left_eye, yaw_left_eye, pitch_right_eye, yaw_right_eye

def face_gaze_3d(face_2d, face_3d, img_w, img_h):
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    cam_matrix, dist_matrix = get_camera_matrix(img_w, img_h)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 1800
    yaw = -angles[1] * 1800

    return pitch, yaw

def eyes_gaze_2d(left_eye_2d, right_eye_2d, left_pupil_2d, right_pupil_2d):

    # print(f"right_eye_2d: {right_eye_2d}")
    # print(f"left_eye_2d: {left_eye_2d}")
    # print(f"left_pupil_2d: {left_pupil_2d}")
    # print(f"right_pupil_2d: {right_pupil_2d}")
    left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
    right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
    left_eye_center = np.mean(left_eye_2d, axis=0)
    right_eye_center = np.mean(right_eye_2d, axis=0)
    left_pupil_2d = np.array(left_pupil_2d, dtype=np.float64)
    right_pupil_2d = np.array(right_pupil_2d, dtype=np.float64)
    # print(f"left_eye_center: {left_eye_center}")
    # print(f"right_eye_center: {right_eye_center}")
    # print(f"left_pupil_2d: {left_pupil_2d}")
    # print(f"right_pupil_2d: {right_pupil_2d}")
    left_eye_displacement = left_pupil_2d - left_eye_center
    right_eye_displacement = right_pupil_2d - right_eye_center
    left_eye_width = np.linalg.norm(left_eye_2d[0] - left_eye_2d[2])
    right_eye_width = np.linalg.norm(right_eye_2d[0] - right_eye_2d[2])
    # print(f"left_eye_displacement: {left_eye_displacement}")
    # print(f"right_eye_displacement: {right_eye_displacement}")
    # print(f"left_eye_width: {left_eye_width}")
    # print(f"right_eye_width: {right_eye_width}")
    pitch_left_eye = left_eye_displacement[1] / left_eye_width * -120
    yaw_left_eye = left_eye_displacement[0] / left_eye_width * -120
    pitch_right_eye = right_eye_displacement[1] / right_eye_width * -120
    yaw_right_eye = right_eye_displacement[0] / right_eye_width * -120

    return pitch_left_eye, yaw_left_eye, pitch_right_eye, yaw_right_eye

cap = cv2.VideoCapture(0)  # Local webcam (index start from 0)

yaw_pitch_points = [33, 263, 1, 61, 291, 199]
right_EAR_points = {160: None, 158: None, 153: None, 163: None, 33: None, 133: None}
left_EAR_pints = {263: None, 384: None, 387: None, 373: None, 381: None, 362: None}

t1, t2, t3, t4 = 0, 0, 0, 0
eye_state = -1
perc_time = 0
threshold_80 = 0.20
threshold_20 = 0.10

scale_v = 1

distraction_threshold = 30  # in degrees
scale_factor = 0.7

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
            time.sleep(1/25)  # Delay to simulate processing time
            for idx, lm in enumerate(face_landmarks.landmark):
                # EAR points
                if idx in right_EAR_points.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    right_EAR_points[idx] = int(lm.x * img_w), int(lm.y * img_h)
                if idx in left_EAR_pints.keys():
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                    left_EAR_pints[idx] = int(lm.x * img_w), int(lm.y * img_h)
                # _______________________________________________________________________________
                # Face direction points
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                # _______________________________________________________________________________
                # Gaze direction points
                if idx == 468:
                    right_pupil_2d = (lm.x * img_w, lm.y * img_h)
                    right_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                if idx == 33 or idx == 145 or idx == 133 or idx == 159:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    right_eye_2d.append([x, y])
                    right_eye_3d.append([x, y, lm.z])
                if idx == 473:
                    left_pupil_2d = (lm.x * img_w, lm.y * img_h)
                    left_pupil_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                if idx == 362 or idx == 374 or idx == 263 or idx == 386:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    left_eye_2d.append([x, y])
                    left_eye_3d.append([x, y, lm.z])

            pitch, yaw = face_gaze_3d(face_2d, face_3d, img_w, img_h)
            pitch_left_eye, yaw_left_eye, pitch_right_eye, yaw_right_eye = eyes_gaze_2d(left_eye_2d, right_eye_2d, left_pupil_2d, right_pupil_2d)

            # fine tuning for pitch and yaw
            pitch -= 20
            yaw += 15
            pitch /= 2
            yaw /= 2
            pitch_left_eye -= 5
            pitch_right_eye -= 5
            pitch_left_eye *= 3
            pitch_right_eye *= 3

            # print(f" Pitch: {pitch}, Yaw: {yaw}")
            # print(f" Pitch Left Eye: {pitch_left_eye}, Yaw Left Eye: {yaw_left_eye}")
            # print(f" Pitch Right Eye: {pitch_right_eye}, Yaw Right Eye: {yaw_right_eye}")
        
            roll = 180 + (np.arctan2(right_EAR_points[33][1] - left_EAR_pints[263][1], right_EAR_points[33][0] - left_EAR_pints[263][0]) * 180 / np.pi)
            if roll > 180:
                roll = roll - 360
            
            # DISPLAY DIRECTIONS
            # Nose direction
            # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] - yaw * scale_factor), int(nose_2d[1] - pitch * scale_factor))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Left eye direction
            # left_pupil_3d_projection, jacobian_left_eye = cv2.projectPoints(left_pupil_3d, rot_vec_left_eye, trans_vec_left_eye, cam_matrix, dist_matrix)
            p1_left_eye = (int(left_pupil_2d[0]), int(left_pupil_2d[1]))
            p2_left_eye = (int(left_pupil_2d[0] - yaw_left_eye * scale_factor), int(left_pupil_2d[1] - pitch_left_eye * scale_factor))
            cv2.line(image, p1_left_eye, p2_left_eye, (0, 0, 255), 3)

            # Right eye direction
            # right_pupil_3d_projection, jacobian_right_eye = cv2.projectPoints(right_pupil_3d, rot_vec_right_eye, trans_vec_right_eye, cam_matrix, dist_matrix)
            p1_right_eye = (int(right_pupil_2d[0]), int(right_pupil_2d[1]))
            p2_right_eye = (int(right_pupil_2d[0] - yaw_right_eye * scale_factor), int(right_pupil_2d[1] - pitch_right_eye * scale_factor))
            cv2.line(image, p1_right_eye, p2_right_eye, (0, 255, 0), 3)

            if PRINT == 1:
                cv2.putText(image, f'P: {float(pitch):.3f}, Y: {float(yaw):.3f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(image, f'PL: {float(pitch_left_eye):.3f}, YL: {float(yaw_left_eye):.3f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(image, f'PR: {float(pitch_right_eye):.3f}, YR: {float(yaw_right_eye):.3f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            if abs(pitch + (pitch_left_eye + pitch_right_eye)/2) > distraction_threshold or abs(yaw + (yaw_left_eye + yaw_right_eye)/2) > distraction_threshold:
                cv2.putText(image, 'Eyes on the road!', (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # EAR computation
            EAR_DX = (
                euclidean_distance(right_EAR_points[160], right_EAR_points[163]) +
                euclidean_distance(right_EAR_points[158], right_EAR_points[153])
            ) / (
                2 * euclidean_distance(right_EAR_points[33], right_EAR_points[133])
            )
            EAR_SX = (
                euclidean_distance(left_EAR_pints[384], left_EAR_pints[381]) +
                euclidean_distance(left_EAR_pints[387], left_EAR_pints[373])
            ) / (
                2 * euclidean_distance(left_EAR_pints[362], left_EAR_pints[263])
            )

            if PRINT == 2:
                cv2.putText(image, f'EAR DX : {float(EAR_DX):.3f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(image, f'EAR SX : {float(EAR_SX):.3f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            if EAR_DX >= threshold_80 and EAR_SX >= threshold_80:
                # eyes open, idle state
                if eye_state != 0:
                    if eye_state == 3:
                        t4 = time.time()
                        perc_time = (t3 - t2) / (t4 - t1)
                    # print(f"eye state 0")
                    eye_state = 0
                    t0 = time.time()
                    t1, t2, t3, t4 = 0, 0, 0, 0

            elif (EAR_DX < threshold_80 and EAR_DX > threshold_20) or (EAR_SX < threshold_80 and EAR_SX > threshold_20):
                if eye_state == 0:
                    # print(f"eye state 1")
                    eye_state = 1
                    t1 = time.time()
                elif eye_state == 2:
                    # print(f"eye state 3")
                    eye_state = 3
                    t3 = time.time()
                else:
                    pass
            elif EAR_DX <= threshold_20 and EAR_SX <= threshold_20:
                if eye_state == 1:
                    # print(f"eye state 2")
                    eye_state = 2
                    t2 = time.time()
                elif eye_state == 0 or eye_state == 3:
                    eye_state = -1
            else :
                pass

    if time.time() - t0 > 10: # In secondi
            cv2.putText(image, f'Blink your eyes!', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    if PRINT == 2:
        cv2.putText(image, f'EYE STATE : {eye_state:.2f}', (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(image, f'PERCLOS : {perc_time:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime if totalTime > 0 else 0

    cv2.putText(image, f'FPS : {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow('Driver Monitoring - PERCLOS Calculation', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit if ESC is pressed
        break

cap.release()
cv2.destroyAllWindows()
