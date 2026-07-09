import cv2
import numpy as np
import os


# --- Entrada del color de la fruta ---
color_mango = input("Coloca el color de la fruta (verde o amarillo): ").strip().lower()
if color_mango not in ['verde', 'amarillo']:
    raise ValueError("Entrada inválida. Solo se acepta 'verde' o 'amarillo'.")


# --- Rangos de segmentación por tipo de mango ---
if color_mango == 'amarillo':
    hsv_rango = ((15, 135, 80), (30, 255, 255))
    lab_b_rango = (150, 200)
    hsi_s_rango = (0.4, 0.8)
else:
    hsv_rango = ((0, 100, 40), (30, 250, 250))
    lab_b_rango = (150, 190)
    hsi_s_rango = (0.3, 1.0)


# --- Funciones de segmentación ---
def segmentar_hsv(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_rango[0], hsv_rango[1])
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mask)
    return resultado


def segmentar_lab(imagen):
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    canal_b = lab[:, :, 2]
    mask = cv2.inRange(canal_b, lab_b_rango[0], lab_b_rango[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mask)
    return resultado


def segmentar_hsi(imagen):
    img = imagen.astype('float32') / 255.0
    b, g, r = cv2.split(img)
    intensidad = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturacion = 1 - (min_rgb / (intensidad + 1e-6))
    s_min, s_max = hsi_s_rango
    mask = np.where((saturacion >= s_min) & (saturacion <= s_max), 255, 0).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mask)
    return resultado


def procesar_imagenes(carpeta_imagenes, carpeta_salida):
    rutas = {
        'HSV': os.path.join(carpeta_salida, 'HSV'),
        'LAB': os.path.join(carpeta_salida, 'LAB'),
        'HSI': os.path.join(carpeta_salida, 'HSI')
    }

    for path in rutas.values():
        os.makedirs(path, exist_ok=True)

    archivos_imagenes = sorted(os.listdir(carpeta_imagenes))

    for nombre_img in archivos_imagenes:
        if nombre_img.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_img = os.path.join(carpeta_imagenes, nombre_img)
            img = cv2.imread(ruta_img)

            if img is None:
                print(f"No se pudo leer la imagen: {nombre_img}")
                continue

            seg_hsv = segmentar_hsv(img)
            seg_lab = segmentar_lab(img)
            seg_hsi = segmentar_hsi(img)

            cv2.imwrite(os.path.join(rutas['HSV'], nombre_img), seg_hsv)
            cv2.imwrite(os.path.join(rutas['LAB'], nombre_img), seg_lab)
            cv2.imwrite(os.path.join(rutas['HSI'], nombre_img), seg_hsi)

    print("Segmentaciones guardadas correctamente.")


# --- Rutas (modifica según tus directorios) ---
carpeta_imagenes = 'ruta/a/tu/carpeta_de_imagenes'
carpeta_salida = 'ruta/a/tu/carpeta_de_salida'

procesar_imagenes(carpeta_imagenes, carpeta_salida)