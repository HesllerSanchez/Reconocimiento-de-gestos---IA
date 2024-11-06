'''
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Puntos de referencia para las cejas y ojos
index_left_eyebrow = [70, 63, 105, 66, 107]
index_right_eyebrow = [336, 296, 334, 293, 300]
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

# Variables para umbral dinámico
initial_distance_left = None
initial_distance_right = None
EYEBROW_RAISE_FACTOR = 1.25  # Factor para establecer el umbral dinámico
NUM_FRAMES_CEJAS = 2  # Frames necesarios para confirmar levantamiento

# Configurar la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Inicializar el color del recuadro
background_color = (255, 255, 255)  # Blanco al inicio

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Variables para almacenar coordenadas
        coordinates_left_eyebrow = []
        coordinates_right_eyebrow = []
        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener puntos de las cejas y ojos
                for index in index_left_eyebrow:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eyebrow.append([x, y])

                for index in index_right_eyebrow:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eyebrow.append([x, y])

                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])

                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])

                # Calcular distancias para detectar levantamiento de cejas
                left_eyebrow_dist = np.mean([coordinates_left_eyebrow[i][1] - coordinates_left_eye[i % len(coordinates_left_eye)][1] for i in range(len(coordinates_left_eyebrow))])
                right_eyebrow_dist = np.mean([coordinates_right_eyebrow[i][1] - coordinates_right_eye[i % len(coordinates_right_eye)][1] for i in range(len(coordinates_right_eyebrow))])

                # Establecer umbral dinámico si es la primera medición
                if initial_distance_left is None and initial_distance_right is None:
                    initial_distance_left = left_eyebrow_dist
                    initial_distance_right = right_eyebrow_dist
                    EYEBROW_RAISE_THRESHOLD_LEFT = initial_distance_left * EYEBROW_RAISE_FACTOR
                    EYEBROW_RAISE_THRESHOLD_RIGHT = initial_distance_right * EYEBROW_RAISE_FACTOR

                # Dibuja las cejas en tiempo real
                for i in range(len(coordinates_left_eyebrow) - 1):
                    cv2.line(frame, tuple(coordinates_left_eyebrow[i]), tuple(coordinates_left_eyebrow[i + 1]), (0, 255, 0), 2)
                    cv2.line(frame, tuple(coordinates_right_eyebrow[i]), tuple(coordinates_right_eyebrow[i + 1]), (0, 255, 0), 2)

                # Detecta si ambas cejas están levantadas
                if left_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_LEFT and right_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_RIGHT:
                    # Cambia el color de fondo a verde cuando se detecta el levantamiento
                    background_color = (0, 255, 0)
                else:
                    # Cambia el color de fondo de regreso a blanco
                    background_color = (255, 255, 255)

        # Mostrar el recuadro en pantalla
        cv2.rectangle(frame, (0, 0), (250, 50), background_color, -1)
        cv2.putText(frame, "ADELANTE (lev. cejas):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key to stop
            break

cap.release()
cv2.destroyAllWindows()

'''

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Puntos de referencia para las cejas y ojos
index_left_eyebrow = [70, 63, 105, 66, 107]
index_right_eyebrow = [336, 296, 334, 293, 300]
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

# Variables para umbral dinámico
initial_distance_left = None
initial_distance_right = None
EYEBROW_RAISE_FACTOR = 1.25  # Factor para establecer el umbral dinámico
NUM_FRAMES_CEJAS = 2  # Frames necesarios para confirmar levantamiento

# Configurar la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Inicializar el color del recuadro
background_color = (255, 255, 255)  # Blanco por default
eyebrow_raised = False
raise_start_time = 0

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Variables para almacenar coordenadas
        coordinates_left_eyebrow = []
        coordinates_right_eyebrow = []
        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener puntos de las cejas y ojos
                for index in index_left_eyebrow:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eyebrow.append([x, y])

                for index in index_right_eyebrow:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eyebrow.append([x, y])

                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])

                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])

                # Calcular distancias para detectar levantamiento de cejas
                left_eyebrow_dist = np.mean([coordinates_left_eyebrow[i][1] - coordinates_left_eye[i % len(coordinates_left_eye)][1] for i in range(len(coordinates_left_eyebrow))])
                right_eyebrow_dist = np.mean([coordinates_right_eyebrow[i][1] - coordinates_right_eye[i % len(coordinates_right_eye)][1] for i in range(len(coordinates_right_eyebrow))])

                # Establecer umbral dinámico si es la primera medición
                if initial_distance_left is None and initial_distance_right is None:
                    initial_distance_left = left_eyebrow_dist
                    initial_distance_right = right_eyebrow_dist
                    EYEBROW_RAISE_THRESHOLD_LEFT = initial_distance_left * EYEBROW_RAISE_FACTOR
                    EYEBROW_RAISE_THRESHOLD_RIGHT = initial_distance_right * EYEBROW_RAISE_FACTOR

                # Dibujar la forma de las cejas
                cv2.fillPoly(frame, [np.array(coordinates_left_eyebrow)], (0, 255, 0))
                cv2.fillPoly(frame, [np.array(coordinates_right_eyebrow)], (0, 255, 0))

                # Detectar si ambas cejas están levantadas
                if left_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_LEFT and right_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_RIGHT:
                    if not eyebrow_raised:
                        eyebrow_raised = True
                        raise_start_time = time.time()
                        background_color = (0, 255, 0)  # Cambia el color a verde
                else:
                    # Verificar si ha pasado 1.5 segundos desde el levantamiento de cejas
                    if eyebrow_raised and (time.time() - raise_start_time > 1.5):
                        eyebrow_raised = False
                        background_color = (255, 255, 255)  # Vuelve a blanco después de 1.5 segundos

        # Mostrar el recuadro en pantalla
        cv2.rectangle(frame, (0, 0), (250, 50), background_color, -1)
        cv2.putText(frame, "ADELANTE (lev. cejas):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key to stop
            break

cap.release()
cv2.destroyAllWindows()

