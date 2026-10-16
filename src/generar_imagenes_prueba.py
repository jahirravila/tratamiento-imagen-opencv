"""
generar_imagenes_prueba.py
--------------------------
Genera el conjunto de 5 imagenes de PRUEBA del mismo objeto (taza de cafe)
bajo condiciones de captura distintas, simuladas con OpenCV a partir de la
imagen de muestra `skimage.data.coffee()` (dominio publico).

IMPORTANTE: la actividad pide fotografias propias. Este script existe para que
el pipeline funcione de extremo a extremo; para la entrega final reemplace los
archivos de la carpeta `imagenes/` por sus fotos (mismos nombres) y vuelva a
ejecutar el notebook.

Uso:
    python src/generar_imagenes_prueba.py --salida imagenes
"""
import argparse
import os

import cv2
import numpy as np
from skimage import data

SEMILLA = 42


def referencia() -> np.ndarray:
    """Imagen base en BGR (convencion de OpenCV)."""
    return cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)


def baja_iluminacion(img: np.ndarray, factor: float = 0.30, gamma: float = 1.30,
                     sigma_ruido: float = 6.0) -> np.ndarray:
    """Subexposicion: reduce la luz, oscurece los medios tonos (gamma>1)
    y agrega ruido gaussiano, como hace un sensor que sube el ISO."""
    rng = np.random.default_rng(SEMILLA)
    x = img.astype(np.float32) / 255.0
    x = np.power(x * factor, gamma)
    x = x * 255.0 + rng.normal(0, sigma_ruido, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def desenfoque_movimiento(img: np.ndarray, longitud: int = 25,
                          angulo: float = 20.0) -> np.ndarray:
    """Desenfoque por movimiento: convolucion con un kernel lineal rotado
    (simula mover la camara durante la exposicion)."""
    k = np.zeros((longitud, longitud), np.float32)
    k[longitud // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((longitud / 2 - 0.5, longitud / 2 - 0.5), angulo, 1.0)
    k = cv2.warpAffine(k, M, (longitud, longitud))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def _fondo_saturado(alto: int, ancho: int) -> np.ndarray:
    """Fondo con muchos bordes, colores y patrones (escena 'desordenada')."""
    rng = np.random.default_rng(SEMILLA)
    f = np.full((alto, ancho, 3), 200, np.uint8)
    for _ in range(140):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        t = rng.integers(0, 4)
        p1 = (int(rng.integers(0, ancho)), int(rng.integers(0, alto)))
        if t == 0:
            p2 = (p1[0] + int(rng.integers(20, 160)), p1[1] + int(rng.integers(20, 120)))
            cv2.rectangle(f, p1, p2, color, -1)
        elif t == 1:
            cv2.circle(f, p1, int(rng.integers(8, 60)), color, -1)
        elif t == 2:
            p2 = (int(rng.integers(0, ancho)), int(rng.integers(0, alto)))
            cv2.line(f, p1, p2, color, int(rng.integers(1, 6)))
        else:
            cv2.putText(f, "OpenCV", p1, cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.5, 1.6)), color, 2)
    # rayas diagonales de alta frecuencia en un tercio de la escena
    for i in range(0, ancho + alto, 9):
        cv2.line(f, (i, 0), (i - alto, alto), (30, 30, 30), 1)
    return cv2.GaussianBlur(f, (3, 3), 0)


def mascara_objeto(img: np.ndarray) -> np.ndarray:
    """Segmenta taza + plato con GrabCut a partir de un rectangulo inicial."""
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (70, 15, 415, 380)  # x, y, ancho, alto que encierra taza y plato
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # conservar solo la componente conexa mas grande
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.where(lab == mayor, 255, 0).astype(np.uint8)
    return cv2.GaussianBlur(m, (5, 5), 0)


def fondo_complejo(img: np.ndarray) -> np.ndarray:
    """Coloca el objeto segmentado sobre un fondo cargado de bordes y colores."""
    alfa = mascara_objeto(img).astype(np.float32)[..., None] / 255.0
    fondo = _fondo_saturado(*img.shape[:2]).astype(np.float32)
    return (alfa * img + (1 - alfa) * fondo).astype(np.uint8)


def cambio_perspectiva(img: np.ndarray) -> np.ndarray:
    """Camara inclinada y rotada: homografia + menor resolucion de salida.
    Las esquinas destino quedan fuera del cuadro para que no aparezcan bordes."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-w * 0.02, -h * 0.30], [w * 1.25, h * 0.05],
                      [w * 1.02, h * 1.30], [-w * 0.30, h * 1.00]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return cv2.resize(out, (480, 320), interpolation=cv2.INTER_AREA)


def main(salida: str) -> None:
    os.makedirs(salida, exist_ok=True)
    ref = referencia()
    q = [cv2.IMWRITE_JPEG_QUALITY, 92]
    cv2.imwrite(os.path.join(salida, "1_referencia.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(salida, "2_baja_iluminacion.jpg"), baja_iluminacion(ref), q)
    cv2.imwrite(os.path.join(salida, "3_desenfoque_movimiento.jpg"), desenfoque_movimiento(ref), q)
    cv2.imwrite(os.path.join(salida, "4_fondo_complejo.jpg"), fondo_complejo(ref), q)
    # se guarda en PNG a proposito para comparar formato/tamano de archivo
    cv2.imwrite(os.path.join(salida, "5_perspectiva.png"), cambio_perspectiva(ref))
    print("Imagenes generadas en", os.path.abspath(salida))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="imagenes")
    main(ap.parse_args().salida)
