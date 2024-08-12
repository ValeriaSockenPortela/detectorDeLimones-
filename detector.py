import cv2
import numpy as np

limonFase1Arriba = np.array([52, 136, 59])
limonFase1Abajo = np.array([49, 132, 55])

limonFase2Arriba = np.array([51, 202, 117])
limonFase2Abajo = np.array([48, 185, 112])

limonFase3Arriba = np.array([43, 187, 142])
limonFase3Abajo = np.array([40, 181, 135])

limonFase4Arriba = np.array([34, 180, 164])
limonFase4Abajo = np.array([32, 167, 160])

# Factor de escala: 1 cm = 50 píxeles (ajusta según tu calibración)
scale_factor = 50.0

# Crear el detector SIFT
sift = cv2.SIFT_create()

# Cargar la imagen de referencia
reference_image_path = 'C:/Users/Socken/Documents/GitHub/detectorDeLimones-/limon.jpg'  # Reemplaza con la ruta de la imagen
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # Cambié de 1 a 0 para usar la cámara predeterminada
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Crear un objeto de coincidencia de características
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Crear una ventana redimensionable
cv2.namedWindow('Frame Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('SIFT Matches', cv2.WINDOW_NORMAL)

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (el stream se ha terminado?). Saliendo ...")
        break

    # Convertir el frame a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Convertir el frame a colores hsv
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
        #print(area_cm2)
        # Dibujar solo si el área es mayor a 20 cm²
        if area_cm2 > 2.0:     #<-------------------------------------------------------------------------------------------------- AQUI
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
            cv2.putText(frame, f'Perimeter: {perimeter_cm:.2f} cm', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            mask = np.zeros_like(gray_frame)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(hsv_frame, mask=mask)[:3]
            #print(mean_color)
            if limonFase1Abajo[0] <= mean_color[0] and limonFase1Abajo[1] <= mean_color[1] and limonFase1Abajo[2] <= mean_color[2]:
                print("Limon de fase 1")
            else:
                print("No coincide")
            


    # Detectar puntos clave y calcular descriptores en el cuadro actual para SIFT
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

    # Encontrar los emparejamientos entre los descriptores de la imagen de referencia y los del cuadro actual
    if descriptors is not None and len(descriptors) > 0:
        matches = bf.match(reference_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Dibujar los primeros 10 emparejamientos
        frame_with_matches = cv2.drawMatches(reference_image, reference_keypoints, frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Mostrar el cuadro con los emparejamientos dibujados
        cv2.imshow('SIFT Matches', frame_with_matches)
    else:
        cv2.imshow('SIFT Matches', frame)  # Mostrar el frame original si no hay emparejamientos

    # Mostrar el frame original con todos los contornos y tamaños
    cv2.imshow('Frame Original', frame)

    # Salimos del bucle con la tecla 'a'
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Liberar el capture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()