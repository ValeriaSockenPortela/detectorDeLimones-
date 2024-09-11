import cv2
import numpy as np
from time import sleep

# Valores en RGB para cada fase

limonFase1Arriba = np.array([27, 98, 70])  
limonFase1Abajo = np.array([7, 78, 50])
limonFase2Arriba = np.array([57, 158, 85])
limonFase2Abajo = np.array([37, 138, 65])
limonFase3Arriba = np.array([228, 255, 233])
limonFase3Abajo = np.array([208, 241, 213])
limonFase4Arriba = np.array([103, 237, 211])
limonFase4Abajo = np.array([83, 217, 191])

scale_factor = 50.0
brillo = 0  # Inicialmente 50
umbralNumero = 37
# Iniciar captura de video
ip = ''
cap = cv2.VideoCapture(ip)  
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

def ajustarBrillo(hsv_frame, brillo):
    h, s, v = cv2.split(hsv_frame)
    v = cv2.add(v, brillo)
    return cv2.merge([h, s, v])

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (el stream se ha terminado?). Saliendo ...")
        break

    # Ajustar el brillo del frame en el espacio HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = ajustarBrillo(hsv_frame, brillo)
    frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    
    # Convertir a escala de grises y aplicar umbralización adaptativa
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, umbralNumero, 2)
    
    # Aplicar suavizado Gaussiano y operación morfológica para reducir ruido
    filtro = cv2.GaussianBlur(umbral, (7, 7), 0)
    kernel = np.ones((3, 3), np.uint8)
    filtro = cv2.morphologyEx(filtro, cv2.MORPH_OPEN, kernel)
    # Detección de bordes con Canny
    bordes = cv2.Canny(filtro, 150, 200)
    contornoActual, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar los contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        areaPixels = cv2.contourArea(contorno)
        area = areaPixels / (scale_factor ** 2)

        # Filtrar contornos pequeños para mayor estabilidad
        if area > 2.0:
            cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)

            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Crear máscara y calcular el color promedio del contorno
            mask = np.zeros_like(gris)
            cv2.drawContours(mask, [contorno], -1, 255, -1)
            color = cv2.mean(frame, mask=mask)[:3]
            print(color)
            fase = detectarFase(color)
            
            # Mostrar resultados en la imagen
            cv2.putText(frame, f'{area:.2f} cm', (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, fase, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            #if fase == 'No coincide':
            #    brillo = brillo +1
    # Mostrar las imágenes procesadas
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Frame en Binario', umbral)
    cv2.imshow('Frame Blanco y Negro', gris)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        brillo = min(brillo + 1, 255)
        print(f'Brillo aumentado a: {brillo}')
    elif key == ord('s'):
        brillo = brillo - 1
        print(f'Brillo reducido a: {brillo}')
    elif key == ord('e'):
        brillo = min(brillo + 10, 255)
        print(f'Brillo aumentado a: {brillo}')
    elif key == ord('d'):
        brillo = brillo - 10
        print(f'Brillo reducido a: {brillo}')
    elif key == ord('r'):
        umbralNumero = umbralNumero +2
        print(f'Umbral aumento: {umbralNumero}')
    elif key == ord('f'):
        umbralNumero = max(umbralNumero -2, 3)
        print(f'Umbral disminullo: {umbralNumero}')
    elif key == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
