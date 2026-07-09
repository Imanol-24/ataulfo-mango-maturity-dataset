import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis, entropy


carpetas = [
    'ruta/a/tu/carpeta/R01',
    'ruta/a/tu/carpeta/R02',
    'ruta/a/tu/carpeta/R03',
    'ruta/a/tu/carpeta/R04'
]

carpeta_nombres = ['R01', 'R02', 'R03', 'R04']
extensiones_validas = ['.jpg', '.jpeg', '.png']
rango_lab_b = {'verde': (150, 190), 'amarillo': (150, 200)}
carpeta_salida = 'ruta/a/tu/carpeta_de_salida'

os.makedirs(carpeta_salida, exist_ok=True)


def eliminar_fondo_negro_lab_color(imagen_segmentada, tipo_mango):
    lab_img = cv2.cvtColor(imagen_segmentada, cv2.COLOR_BGR2Lab)
    b_min, b_max = rango_lab_b[tipo_mango]
    b_mask = cv2.inRange(lab_img[:, :, 2], b_min, b_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_refinada = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel)
    return mascara_refinada


def min_rotation_pattern(pattern, p):
    min_val = pattern
    for i in range(1, p):
        rotated = ((pattern >> i) | ((pattern & ((1 << i) - 1)) << (p - i))) & ((1 << p) - 1)
        if rotated < min_val:
            min_val = rotated
    return min_val


def lbp_riu2(lbp_img, p):
    uniform_codes = {}
    code_counter = 0

    for pattern in range(0, 1 << p):
        transitions = 0
        prev_bit = pattern & 1

        for i in range(1, p + 1):
            current_bit = (pattern >> (i % p)) & 1
            if prev_bit != current_bit:
                transitions += 1
                prev_bit = current_bit
                if transitions > 2:
                    break

        if transitions <= 2:
            min_rot = min_rotation_pattern(pattern, p)
            if min_rot not in uniform_codes:
                uniform_codes[min_rot] = code_counter
                code_counter += 1

    riu2_lbp = np.zeros_like(lbp_img, dtype=np.uint8)

    for i in range(lbp_img.shape[0]):
        for j in range(lbp_img.shape[1]):
            pattern = int(lbp_img[i, j])
            if pattern in uniform_codes:
                riu2_lbp[i, j] = uniform_codes[pattern]
            else:
                riu2_lbp[i, j] = code_counter

    return riu2_lbp, code_counter + 1


def calcular_histograma_lbpriu2(img_gray, mask, p=8, r=1):
    lbp_basic = local_binary_pattern(img_gray, p, r, method='default')
    lbp_riu2_img, n_bins = lbp_riu2(lbp_basic, p)
    lbp_masked = lbp_riu2_img[mask > 0]
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def calcular_descriptores_histograma(hist):
    mean = np.mean(hist)
    ent = entropy(hist + 1e-10)
    skewness = skew(hist)
    kurt = kurtosis(hist)
    std = np.std(hist)
    energy = np.sum(hist ** 2)
    return mean, ent, skewness, kurt, std, energy


print("Selecciona el directorio a procesar:")
for idx, name in enumerate(carpeta_nombres):
    print(f"{idx + 1}. {name} ({carpetas[idx]})")

while True:
    seleccion = input(f"Ingrese el número [1-{len(carpeta_nombres)}]: ")
    try:
        seleccion = int(seleccion)
        if 1 <= seleccion <= len(carpeta_nombres):
            break
        else:
            print("Valor fuera de rango.")
    except:
        print("Por favor, ingrese un número válido.")


directorio = carpetas[seleccion - 1]
nombre_directorio = carpeta_nombres[seleccion - 1]
tipo_mango = 'verde' if seleccion - 1 < 2 else 'amarillo'

print(f"\nProcesando directorio: {directorio}\n")

resultados = []
imagenes = sorted([
    f for f in os.listdir(directorio)
    if os.path.splitext(f)[1].lower() in extensiones_validas
])

for nombre_img in imagenes:
    ruta = os.path.join(directorio, nombre_img)
    img = cv2.imread(ruta)

    if img is None:
        print(f"Error al leer la imagen: {nombre_img}")
        continue

    mask = eliminar_fondo_negro_lab_color(img, tipo_mango)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.sum(mask > 0) == 0:
        print(f"Máscara vacía en la imagen: {nombre_img}")
        continue

    try:
        hist_lbp = calcular_histograma_lbpriu2(img_gray, mask, p=8, r=1)
        mean, ent, skewness, kurt, std, energy = calcular_descriptores_histograma(hist_lbp)

        resultados.append({
            'imagen': nombre_img,
            'mean': mean,
            'entropy': ent,
            'skewness': skewness,
            'kurtosis': kurt,
            'std': std,
            'energy': energy
        })

        print(f"Procesada correctamente: {nombre_img}")

    except Exception as e:
        print(f"Error al procesar {nombre_img}: {str(e)}")


if resultados:
    df = pd.DataFrame(resultados)
    csv_path = os.path.join(carpeta_salida, f'lbpriu2_descriptores_{nombre_directorio}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")
    print(f"Total de imágenes procesadas: {len(resultados)}")
else:
    print("No se procesaron imágenes. Verifique los datos de entrada.")