
import cv2
import mediapipe as mp
import numpy as np

def drawing_output(frame, left_corner, right_corner, is_smiling, mouth_points):
    # Cambiar color del recuadro según el estado de la sonrisa
    color = (0, 255, 0) if is_smiling else (255, 255, 255)  # Verde si está sonriendo, blanco si no

    # Mostrar el cuadro en pantalla con "DETENER"
    cv2.rectangle(frame, (0, 0), (200, 50), color, -1)
    cv2.putText(frame, "DETENER(sonria)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Dibujar la forma de la boca
    for i in range(len(mouth_points) - 1):
        cv2.line(frame, mouth_points[i], mouth_points[i + 1], (0, 255, 0), 2)
    cv2.line(frame, mouth_points[-1], mouth_points[0], (0, 255, 0), 2)  # Cerrar el contorno de la boca

    return frame

def mouth_aspect_ratio(left_corner, right_corner):
    return np.linalg.norm(np.array(left_corner) - np.array(right_corner))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
index_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]  # Índices para el contorno de la boca
SMILE_FRAMES = 5  # Número de frames para considerar una sonrisa
aux_smile_counter = 0
smile_threshold = None  # Umbral dinámico para la sonrisa

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        is_smiling = False  # Indicador de sonrisa en el frame actual

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener las coordenadas de las comisuras de la boca
                mouth_points = [(int(face_landmarks.landmark[i].x * width), 
                                 int(face_landmarks.landmark[i].y * height)) for i in index_mouth]
                left_corner = mouth_points[0]
                right_corner = mouth_points[5]

                # Calcular la distancia entre las comisuras
                mouth_distance = mouth_aspect_ratio(left_corner, right_corner)
                
                # Establecer el umbral en la primera captura neutral de la boca
                if smile_threshold is None:
                    smile_threshold = mouth_distance * 1.05  # Ajuste del umbral según la necesidad

                # Detectar sonrisa si la distancia supera el umbral
                if mouth_distance > smile_threshold:
                    aux_smile_counter += 1
                else:
                    if aux_smile_counter >= SMILE_FRAMES:
                        is_smiling = True  # Activar el cambio de color momentáneo
                    aux_smile_counter = 0

                # Mostrar el cuadro "DETENER" y la forma de la boca
                frame = drawing_output(frame, left_corner, right_corner, is_smiling, mouth_points)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Presiona 'Esc' para salir
            break

cap.release()
cv2.destroyAllWindows()
