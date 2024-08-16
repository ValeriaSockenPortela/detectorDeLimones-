import cv2
import numpy as np
from time import sleep

limonFase1Arriba = np.array([56, 172, 85])
limonFase1Abajo = np.array([34, 149, 64])

limonFase2Arriba = np.array([55, 219, 146])
limonFase2Abajo = np.array([34, 197, 126])

limonFase3Arriba = np.array([46, 220, 167])
limonFase3Abajo = np.array([29, 199, 146])

limonFase4Arriba = np.array([40, 206, 193])
limonFase4Abajo = np.array([20, 185, 173])

limonFase5Arriba = np.array([37, 53, 93])
limonFase5Abajo = np.array([10, 32, 52])

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

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (el stream se ha terminado?). Saliendo ...")
        break

    # Convertir el frame a escala de grises y a colores HSV
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Aplicar un filtro de suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Detección de bordes usando el detector de Canny
    edges = cv2.Canny(blurred, 50, 150)

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
        if area_cm2 > 2.0:
            # Dibujar el contorno en la imagen original
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Obtener el centro del contorno para colocar el texto
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Imprimir el área y el perímetro en centímetros
            cv2.putText(frame, f'Area: {area_cm2:.2f} cm^2', (cX, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #cv2.putText(frame, f'Perimeter: {perimeter_cm:.2f} cm', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Crear una máscara del contorno para calcular el color promedio
            mask = np.zeros_like(gray_frame)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(hsv_frame, mask=mask)[:3]
            print(mean_color)
            sleep(1)
            # Clasificación según el color promedio en el espacio HSV
            if limonFase1Abajo[0] <= mean_color[0] <= limonFase1Arriba[0] and limonFase1Abajo[1] <= mean_color[1] <= limonFase1Arriba[1] and limonFase1Abajo[2] <= mean_color[2] <= limonFase1Arriba[2]:
                print("Limon de fase 1")
            elif limonFase2Abajo[0] <= mean_color[0] <= limonFase2Arriba[0] and limonFase2Abajo[1] <= mean_color[1] <= limonFase2Arriba[1] and limonFase2Abajo[2] <= mean_color[2] <= limonFase2Arriba[2]:
                print("Limon de fase 2")
            elif limonFase3Abajo[0] <= mean_color[0] <= limonFase3Arriba[0] and limonFase3Abajo[1] <= mean_color[1] <= limonFase3Arriba[1] and limonFase3Abajo[2] <= mean_color[2] <= limonFase3Arriba[2]:
                print("Limon de fase 3")
            elif limonFase4Abajo[0] <= mean_color[0] <= limonFase4Arriba[0] and limonFase4Abajo[1] <= mean_color[1] <= limonFase4Arriba[1] and limonFase4Abajo[2] <= mean_color[2] <= limonFase4Arriba[2]:
                print("Limon de fase 4")
            elif limonFase5Abajo[0] <= mean_color[0] <= limonFase5Arriba[0] and limonFase5Abajo[1] <= mean_color[1] <= limonFase5Arriba[1] and limonFase5Abajo[2] <=mean_color[2] <= limonFase1Arriba[2]:
                print("Limon de fase 5")
            else:
                print("No coincide")

    # Mostrar el frame original con los contornos
    cv2.imshow('Frame Original', frame)

    # Salimos del bucle con la tecla 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Liberar el capture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
