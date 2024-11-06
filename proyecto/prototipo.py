import cv2
import mediapipe as mp
import numpy as np

import time

# Inicialización de MediaPipe y configuración de índices
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Puntos de referencia para las cejas, ojos y boca
index_left_eyebrow = [70, 63, 105, 66, 107]
index_right_eyebrow = [336, 296, 334, 293, 300]
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
index_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Umbrales y variables de control
EYEBROW_RAISE_FACTOR = 1.25
SMILE_FRAMES = 5
EAR_THRESH = 0.26
GREEN_DURATION = 1.5  # Tiempo en segundos para mantener el color verde

# Variables para umbrales dinámicos
initial_distance_left = None
initial_distance_right = None
smile_threshold = None  # Umbral dinámico para la sonrisa
aux_smile_counter = 0

# Variables para temporizar el color verde de los cuadros
right_start_time = None
left_start_time = None


# Función para dibujar los cuadros de estado en la pantalla
def drawing_output(frame, move_left, move_right, move_up, is_smiling, mouth_points):
    # Cuadro para el ojo derecho
    right_color = (0, 255, 0) if move_right else (255, 255, 255)
    cv2.rectangle(frame, (0, 0), (150, 50), right_color, -1)
    cv2.putText(frame, "Derecha", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Cuadro para el ojo izquierdo
    left_color = (0, 255, 0) if move_left else (255, 255, 255)
    cv2.rectangle(frame, (0, 60), (150, 110), left_color, -1)
    cv2.putText(frame, "Izquierda", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Cuadro para la sonrisa
    smile_color = (0, 255, 0) if is_smiling else (255, 255, 255) #Cambiar a color verde si está sonriendo
    cv2.rectangle(frame, (0, 120), (150, 170), smile_color, -1)
    cv2.putText(frame, "DETENER (sonría)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Dibujar la forma de la boca
    for i in range(len(mouth_points) - 1):
        cv2.line(frame, mouth_points[i], mouth_points[i + 1], (0, 255, 0), 2)
    cv2.line(frame, mouth_points[-1], mouth_points[0], (0, 255, 0), 2)  # Cerrar el contorno de la boca


    return frame

# Función para calcular el índice de aspecto de la boca
def mouth_aspect_ratio(left_corner, right_corner):
    return np.linalg.norm(np.array(left_corner) - np.array(right_corner))

# Función para calcular el índice de aspecto del ojo
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

# Configurar la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
        mouth_points = []

        move_left = move_right = move_up = is_smiling = False

        if results.multi_face_landmarks:
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

                for i in index_mouth:
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    mouth_points.append((x, y))

                # Calcular distancias para detectar levantamiento de cejas
                left_eyebrow_dist = np.mean([coordinates_left_eyebrow[i][1] - coordinates_left_eye[i % len(coordinates_left_eye)][1] for i in range(len(coordinates_left_eyebrow))])
                right_eyebrow_dist = np.mean([coordinates_right_eyebrow[i][1] - coordinates_right_eye[i % len(coordinates_right_eye)][1] for i in range(len(coordinates_right_eyebrow))])

                # Establecer umbral dinámico si es la primera medición
                if initial_distance_left is None and initial_distance_right is None:
                    initial_distance_left = left_eyebrow_dist
                    initial_distance_right = right_eyebrow_dist
                    EYEBROW_RAISE_THRESHOLD_LEFT = initial_distance_left * EYEBROW_RAISE_FACTOR
                    EYEBROW_RAISE_THRESHOLD_RIGHT = initial_distance_right * EYEBROW_RAISE_FACTOR

                # Detectar movimiento de cejas
                move_up = left_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_LEFT and right_eyebrow_dist > EYEBROW_RAISE_THRESHOLD_RIGHT

                # Detectar movimiento en los ojos
                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)

                if ear_right_eye < EAR_THRESH:
                    move_right = True
                    right_start_time = time.time()
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

                # Calcular la distancia entre las comisuras de la boca
                left_corner = mouth_points[0]
                right_corner = mouth_points[5]
                mouth_distance = mouth_aspect_ratio(left_corner, right_corner)

                # Establecer el umbral en la primera captura neutral de la boca
                if smile_threshold is None:
                    smile_threshold = mouth_distance * 1.25

                # Detectar sonrisa
                if mouth_distance > smile_threshold:
                    aux_smile_counter += 1
                else:
                    if aux_smile_counter >= SMILE_FRAMES:
                        is_smiling = True
                    aux_smile_counter = 0

        # Dibujar los cuadros de estado
        frame = drawing_output(frame, move_left, move_right, move_up, is_smiling, mouth_points)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
