import cv2
import numpy as np

# Factor de escala: 1 cm = 50 píxeles (ajusta según tu calibración)
scale_factor = 50.0

# Crear el detector SIFT
sift = cv2.SIFT_create()

# Cargar la imagen de referencia
reference_image_path = 'C:/Users/Socken/Documents/Python/Detector de Limones 1.0/limon.jpg'  # Reemplaza con la ruta de tu imagen de referencia
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)
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

        # Dibujar el contorno en la imagen original
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # Obtener el centro del contorno para colocar el texto
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Convertir el tamaño del contorno de píxeles a centímetros
        area_cm2 = area_pixels / (scale_factor ** 2)
        perimeter_cm = perimeter_pixels / scale_factor

        # Imprimir el área y el perímetro en centímetros
        cv2.putText(frame, f'Area: {area_cm2:.2f} cm^2', (cX, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f'Perimeter: {perimeter_cm:.2f} cm', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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

    # Salimos del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el capture y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
