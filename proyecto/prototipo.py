import cv2
import mediapipe as mp
import numpy as np
import time

# Configuración de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Malla en verde

# Índices para la detección de gestos
index_eyebrows_left = [70, 63, 105, 66]
index_eyebrows_right = [336, 296, 334, 293]
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
index_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]

cap = cv2.VideoCapture(0)

# Variables de referencia de calibración
baseline_eyebrow_height_left = baseline_eyebrow_height_right = None
baseline_ear_left = baseline_ear_right = baseline_mar = None

def calculate_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

def calibrate_baseline(face_landmarks, width, height):
    global baseline_ear_left, baseline_ear_right, baseline_mar
    global baseline_eyebrow_height_left, baseline_eyebrow_height_right

    # Obtener coordenadas para ojos, boca y cejas
    left_eye_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_left_eye]
    right_eye_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_right_eye]
    mouth_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_mouth]
    eyebrow_left_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_eyebrows_left]
    eyebrow_right_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_eyebrows_right]

    # Calcular los aspect ratios de los ojos y la boca
    baseline_ear_left = calculate_aspect_ratio(left_eye_coords)
    baseline_ear_right = calculate_aspect_ratio(right_eye_coords)
    mouth_width = np.linalg.norm(np.array(mouth_coords[0]) - np.array(mouth_coords[6]))
    mouth_height = np.linalg.norm(np.array(mouth_coords[3]) - np.array(mouth_coords[9]))
    baseline_mar = mouth_height / mouth_width

    # Calcular distancia entre cejas y ojos para las cejas como baseline
    baseline_eyebrow_height_left = np.linalg.norm(np.array(eyebrow_left_coords[0]) - np.array(left_eye_coords[1]))
    baseline_eyebrow_height_right = np.linalg.norm(np.array(eyebrow_right_coords[0]) - np.array(right_eye_coords[1]))

# Función para dibujar las acciones
def drawing_output(frame, forward, move_right, move_left, stop):
    if forward:
        cv2.putText(frame, "Avanzar", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if move_right:
        cv2.putText(frame, "Derecha", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if move_left:
        cv2.putText(frame, "Izquierda", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if stop:
        cv2.putText(frame, "Frenar", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

# Inicio de la calibración y bucle de captura de video
calibrated = False
calibration_time = 3  # Tiempo de calibración en segundos
start_time = time.time()

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        forward = move_right = move_left = stop = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Malla de puntos futurista
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec
                )

                # Calibración inicial
                if not calibrated:
                    elapsed_time = time.time() - start_time
                    cv2.putText(frame, f"Calibrando... {int(calibration_time - elapsed_time)}s", (10, height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if elapsed_time >= calibration_time:
                        calibrate_baseline(face_landmarks, width, height)
                        calibrated = True
                    continue

                # Obtener coordenadas
                left_eye_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_left_eye]
                right_eye_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_right_eye]
                mouth_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_mouth]
                eyebrow_left_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_eyebrows_left]
                eyebrow_right_coords = [[int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)] for idx in index_eyebrows_right]

                # Mostrar en color claro ojos, cejas y boca
                for x, y in left_eye_coords + right_eye_coords + mouth_coords + eyebrow_left_coords + eyebrow_right_coords:
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

                # Calcular ratios en tiempo real y compararlos con el baseline
                ear_left_eye = calculate_aspect_ratio(left_eye_coords)
                ear_right_eye = calculate_aspect_ratio(right_eye_coords)
                eyebrow_height_left = np.linalg.norm(np.array(eyebrow_left_coords[0]) - np.array(left_eye_coords[1]))
                eyebrow_height_right = np.linalg.norm(np.array(eyebrow_right_coords[0]) - np.array(right_eye_coords[1]))
                mouth_width = np.linalg.norm(np.array(mouth_coords[0]) - np.array(mouth_coords[6]))
                mouth_height = np.linalg.norm(np.array(mouth_coords[3]) - np.array(mouth_coords[9]))
                mouth_aspect_ratio = mouth_height / mouth_width

                # Detección de gestos basada en comparación con baseline
                if eyebrow_height_left > baseline_eyebrow_height_left * 1.2 and eyebrow_height_right > baseline_eyebrow_height_right * 1.2:
                    forward = True
                if ear_left_eye < baseline_ear_left * 0.75 and ear_right_eye >= baseline_ear_right * 0.75:
                    move_left = True
                if ear_right_eye < baseline_ear_right * 0.75 and ear_left_eye >= baseline_ear_left * 0.75:
                    move_right = True
                if mouth_width > baseline_mar * 55:  # Usa un factor de aumento adecuado al observar cambios
                    stop = True

        # Mostrar las acciones detectadas
        frame = drawing_output(frame, forward, move_right, move_left, stop)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()