'''
import cv2
import mediapipe as mp
import numpy as np

def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter_left, blink_counter_right):
    aux_image = np.zeros(frame.shape, np.uint8)
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(0, 0, 255))
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

    # Display blink counters for left and right eyes
    cv2.rectangle(output, (0, 0), (250, 100), (255, 0, 0), -1)
    cv2.putText(output, "Parpadeos Ojo Izq:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter_left), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 250), 2)

    cv2.putText(output, "Parpadeos Ojo Der:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "{}".format(blink_counter_right), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 250), 2)
    
    return output

def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26
NUM_FRAMES = 2
aux_counter_left = 0
aux_counter_right = 0
blink_counter_left = 0
blink_counter_right = 0

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)

                # Calculate EAR for each eye independently
                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)

                # Check if left eye is closed
                if ear_left_eye < EAR_THRESH:
                    aux_counter_left += 1
                else:
                    if aux_counter_left >= NUM_FRAMES:
                        blink_counter_left += 1
                    aux_counter_left = 0

                # Check if right eye is closed
                if ear_right_eye < EAR_THRESH:
                    aux_counter_right += 1
                else:
                    if aux_counter_right >= NUM_FRAMES:
                        blink_counter_right += 1
                    aux_counter_right = 0

                frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter_left, blink_counter_right)
        
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()

'''

import cv2
import mediapipe as mp
import numpy as np
import time

def drawing_output(frame, move_right, move_left):
    # Cuadro para el ojo derecho
    right_color = (0, 255, 0) if move_right else (255, 255, 255)
    cv2.rectangle(frame, (0, 0), (150, 50), right_color, -1)
    cv2.putText(frame, "Derecha", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Cuadro para el ojo izquierdo
    left_color = (0, 255, 0) if move_left else (255, 255, 255)
    cv2.rectangle(frame, (0, 60), (150, 110), left_color, -1)
    cv2.putText(frame, "Izquierda", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.26

# Variables para temporizar el color verde de los cuadros
right_start_time = None
left_start_time = None
GREEN_DURATION = 1.5  # Tiempo en segundos para mantener el color verde

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []

        move_right = False
        move_left = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)

                # Calcular EAR para cada ojo
                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)

                # Detectar movimiento en el ojo derecho
                if ear_right_eye < EAR_THRESH:
                    move_right = True
                    right_start_time = time.time()
                
                # Detectar movimiento en el ojo izquierdo
                if ear_left_eye < EAR_THRESH:
                    move_left = True
                    left_start_time = time.time()

        # Cambiar el color del cuadro a verde durante GREEN_DURATION segundos después de detectar movimiento
        if right_start_time and (time.time() - right_start_time < GREEN_DURATION):
            move_right = True
        else:
            move_right = False

        if left_start_time and (time.time() - left_start_time < GREEN_DURATION):
            move_left = True
        else:
            move_left = False

        # Dibujar los cuadros en pantalla
        frame = drawing_output(frame, move_right, move_left)
        
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Presiona 'Esc' para salir
            break

cap.release()
cv2.destroyAllWindows()
