import cv2
import numpy as np
import threading

# Valores en RGB para cada fase
limonFase1Arriba = np.array([55, 139, 109])
limonFase1Abajo = np.array([35, 129, 89])
limonFase2Arriba = np.array([57, 158, 85])
limonFase2Abajo = np.array([37, 138, 65])
limonFase3Arriba = np.array([228, 255, 233])
limonFase3Abajo = np.array([208, 241, 213])
limonFase4Arriba = np.array([103, 237, 211])
limonFase4Abajo = np.array([83, 217, 191])

scale_factor = 50.0
brillo = 0
umbralNumero = 31

# Iniciar captura de video para dos cámaras IP
ip1 = 'rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?real_stream'
ip2 = 'rtsp://192.168.1.11:554/user=admin&password=&channel=1&stream=0.sdp?real_stream'

cap1 = cv2.VideoCapture(ip1)
cap2 = cv2.VideoCapture(ip2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: No se puede abrir una o ambas cámaras")
    exit()

# Función para ajustar el brillo de un frame
def ajustarBrillo(hsv_frame, brillo):
    h, s, v = cv2.split(hsv_frame)
    v = cv2.add(v, brillo)
    return cv2.merge([h, s, v])

# Función para detectar la fase según el color
def detectarFase(mean_color):
    if np.all(limonFase1Abajo <= mean_color) and np.all(mean_color <= limonFase1Arriba):
        return 'Fase 1'
    elif np.all(limonFase2Abajo <= mean_color) and np.all(mean_color <= limonFase2Arriba):
        return 'Fase 2'
    elif np.all(limonFase3Abajo <= mean_color) and np.all(mean_color <= limonFase3Arriba):
        return 'Fase 3'
    elif np.all(limonFase4Abajo <= mean_color) and np.all(mean_color <= limonFase4Arriba):
        return 'Fase 4'
    else:
        return 'No coincide'

# Función para procesar video de cada cámara
def procesar_camara(cap, window_name):
    global brillo, umbralNumero

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"No se puede recibir frame de la cámara {window_name}. Saliendo...")
            break

        # Procesamiento del frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = ajustarBrillo(hsv_frame, brillo)
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, umbralNumero, 2)

        filtro = cv2.GaussianBlur(umbral, (7, 7), 0)
        filtro = cv2.morphologyEx(filtro, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        bordes = cv2.Canny(filtro, 150, 200)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            areaPixels = cv2.contourArea(contorno)
            area = areaPixels / (scale_factor ** 2)

            if area > 2.0:
                cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
                M = cv2.moments(contorno)
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                mask = np.zeros_like(gris)
                cv2.drawContours(mask, [contorno], -1, 255, -1)
                color = cv2.mean(frame, mask=mask)[:3]
                fase = detectarFase(color)
                cv2.putText(frame, f'{area:.2f} cm', (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, fase, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        # Mostrar la imagen procesada
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            brillo = min(brillo + 1, 255)
            print(f'Brillo aumentado a: {brillo}')
        elif key == ord('s'):
            brillo = brillo - 1
            print(f'Brillo reducido a: {brillo}')
        elif key == ord('a'):
            break

# Crear hilos para cada cámara
thread1 = threading.Thread(target=procesar_camara, args=(cap1, 'Cámara 1 - Frame Original'))
thread2 = threading.Thread(target=procesar_camara, args=(cap2, 'Cámara 2 - Frame Original'))

# Iniciar los hilos
thread1.start()
thread2.start()

# Esperar a que ambos hilos terminen
thread1.join()
thread2.join()

cap1.release()
cap2.release()
cv2.destroyAllWindows()