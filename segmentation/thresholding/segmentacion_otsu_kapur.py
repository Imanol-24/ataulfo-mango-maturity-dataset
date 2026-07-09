import os
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.exposure import histogram


def kapur_threshold(grayscale_img):
    hist, bin_centers = histogram(grayscale_img, nbins=256)
    hist = hist / hist.sum()
    cumsum = np.cumsum(hist)
    entropy_bg = np.cumsum(hist * np.log(hist + 1e-10))
    entropy_fg = np.cumsum((hist[::-1] * np.log(hist[::-1] + 1e-10)))[::-1]
    entropy_total = entropy_bg + entropy_fg

    min_t = 5
    max_t = 250

    if max_t > len(entropy_total):
        max_t = len(entropy_total)

    if min_t >= max_t:
        return 128

    threshold = np.argmax(entropy_total[min_t:max_t]) + min_t
    return threshold


def segmentar_imagen(imagen, metodo='otsu'):
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    base = lab[:, :, 2]

    if metodo == 'otsu':
        t = threshold_otsu(base)
    elif metodo == 'kapur':
        t = kapur_threshold(base)
    else:
        raise ValueError("Método no reconocido. Usa 'otsu' o 'kapur'.")

    mascara_binaria = (base > t).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_refinada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel)
    mascara_refinada = cv2.morphologyEx(mascara_refinada, cv2.MORPH_OPEN, kernel)

    return mascara_refinada


def aplicar_mascara(imagen, mascara):
    return cv2.bitwise_and(imagen, imagen, mask=mascara)


def procesar_segmentacion(origen, destino, metodo):
    os.makedirs(destino, exist_ok=True)

    archivos_img = sorted([
        f for f in os.listdir(origen)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for archivo in archivos_img:
        ruta_img = os.path.join(origen, archivo).replace("\\", "/")
        imagen = cv2.imread(ruta_img)

        if imagen is None:
            print(f"[AVISO] No se pudo cargar: {archivo}")
            continue

        mascara_pred = segmentar_imagen(imagen, metodo)
        resultado = aplicar_mascara(imagen, mascara_pred)

        nombre_base = os.path.splitext(archivo)[0]
        nombre_segmentado = f"{nombre_base}_{metodo}_segmentado.png"
        path_segmentado = os.path.join(destino, nombre_segmentado).replace("\\", "/")

        cv2.imwrite(path_segmentado, resultado)

    print(f"Segmentación con {metodo} completada correctamente.")


origen = 'ruta/a/tu/carpeta_de_imagenes'
destino = 'ruta/a/tu/carpeta_de_salida'

print("\nMétodos disponibles:")
print("1. otsu")
print("2. kapur")
metodo = input("Elige el método de segmentación ('otsu' o 'kapur'): ").strip().lower()

if metodo not in ["otsu", "kapur"]:
    print("Método inválido. Usa 'otsu' o 'kapur'.")
else:
    procesar_segmentacion(origen, destino, metodo)