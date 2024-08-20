import cv2
import numpy as np
from time import sleep

# Valores en RGB para cada fase

limonFase1Arriba = np.array([52, 106, 64])  # Nota: RGB se especifica como [R, G, B]
limonFase1Abajo = np.array([32, 86, 44])

limonFase2Arriba = np.array([57, 158, 85])
limonFase2Abajo = np.array([37, 138, 65])

limonFase3Arriba = np.array([228, 255, 233])
limonFase3Abajo = np.array([208, 241, 213])

limonFase4Arriba = np.array([103, 237, 211])
limonFase4Abajo = np.array([83, 217, 191])

# Factor de escala: 1 cm = 50 píxeles
scale_factor = 50.0
#ip de la camara
ip = 'rtsp://192.168.1.103:554/cam/realmonitor?channel=1&subtype=0&authbasic=YWRtaW46QXJ0dXJpdDAu'

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # Cámara predeterminada
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Crear una ventana redimensionable
cv2.namedWindow('Frame Original', cv2.WINDOW_NORMAL)
sleep(2)
while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (el stream se ha terminado?). Saliendo ...")
        break

    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray_frame, (15, 15), 0)
    _, binary_frame = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Detección de bordes usando el detector de Canny
    edges = cv2.Canny(binary_frame, 50, 150)

    # Encontrar todos los contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre todos los contornos detectados
    for contour in contours:
        area_pixels = cv2.contourArea(contour)
        perimeter_pixels = cv2.arcLength(contour, True)

        # Convertir el tamaño del contorno de píxeles a centímetros
        area_cm2 = area_pixels / (scale_factor ** 2)
        perimeter_cm = perimeter_pixels / scale_factor

        # Dibujar solo si el área es mayor a 20 cm²
        if area_cm2 > 0.2:
            # Dibujar el contorno en la imagen original
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Obtener el centro del contorno para colocar el texto
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Crear una máscara del contorno para calcular el color promedio
            mask = np.zeros_like(gray_frame)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]
            print(mean_color)
            #cv2.putText(frame, f'Color{mean_color}.', (cX, cY-30), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1)
            
            # Clasificación según el color promedio en el espacio RGB
            if np.all(limonFase1Abajo <= mean_color) and np.all(mean_color <= limonFase1Arriba):
                cv2.putText(frame, f'Fase1', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            elif np.all(limonFase2Abajo <= mean_color) and np.all(mean_color <= limonFase2Arriba):
                cv2.putText(frame, f'Fase2', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            elif np.all(limonFase3Abajo <= mean_color) and np.all(mean_color <= limonFase3Arriba):
                cv2.putText(frame, f'Fase3', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            elif np.all(limonFase4Abajo <= mean_color) and np.all(mean_color <= limonFase4Arriba):
                cv2.putText(frame, f'Fase4', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            else:
                cv2.putText(frame, f'No coincide', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    # Mostrar el frame original con los contornos
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Frame en Binario', binary_frame)
    cv2.imshow('Frame Blanco y Negro', gray_frame)

    # Salimos del bucle con la tecla 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Liberar el capture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
