"""
diagnostico.py
--------------
Metricas de calidad sin referencia para una imagen (resolucion, canales,
brillo, contraste, nitidez, ruido, bordes, formato y tamano de archivo) y
reglas simples para etiquetar problemas de captura.

Uso:
    python src/diagnostico.py --carpeta imagenes --salida resultados
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd


def entropia(gris: np.ndarray) -> float:
    """Entropia de Shannon del histograma (bits): cantidad de informacion tonal."""
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def nitidez_laplaciano(gris: np.ndarray) -> float:
    """Varianza del Laplaciano: alta = muchos bordes finos = imagen nitida."""
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def nitidez_tenengrad(gris: np.ndarray) -> float:
    """Media de la magnitud del gradiente de Sobel (robusta al ruido aislado)."""
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def ruido_immerkaer(gris: np.ndarray) -> float:
    """Desviacion estandar del ruido (metodo rapido de Immerkaer, 1996):
    un kernel que anula zonas planas y rampas deja pasar sobre todo el ruido."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = gris.shape
    conv = cv2.filter2D(gris.astype(np.float64), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def densidad_bordes(gris: np.ndarray) -> float:
    """Porcentaje de pixeles marcados como borde por Canny (umbrales automaticos)."""
    med = np.median(gris)
    bajo, alto = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    bordes = cv2.Canny(gris, bajo, max(alto, bajo + 1))
    return float((bordes > 0).mean() * 100)


def diagnosticar(ruta: str) -> dict:
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    alto, ancho = img.shape[:2]
    canales = 1 if img.ndim == 2 else img.shape[2]
    bgr = img if canales == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tam_kb = os.path.getsize(ruta) / 1024
    p2, p98 = np.percentile(gris, [2, 98])
    return {
        "imagen": os.path.basename(ruta),
        "formato": os.path.splitext(ruta)[1][1:].upper(),
        "resolucion": f"{ancho}x{alto}",
        "megapixeles": round(ancho * alto / 1e6, 3),
        "canales": canales,
        "tamano_kb": round(tam_kb, 1),
        "bits_por_pixel": round(tam_kb * 1024 * 8 / (ancho * alto), 2),
        "brillo_media": round(float(gris.mean()), 1),
        "pct_oscuros": round(float((gris < 30).mean() * 100), 1),
        "pct_saturados": round(float((gris > 245).mean() * 100), 1),
        "contraste_rms": round(float(gris.std()), 1),
        "rango_dinamico_p2_p98": int(p98 - p2),
        "entropia_bits": round(entropia(gris), 2),
        "nitidez_lap_var": round(nitidez_laplaciano(gris), 1),
        "nitidez_tenengrad": round(nitidez_tenengrad(gris), 1),
        "ruido_sigma": round(ruido_immerkaer(gris), 2),
        "ruido_relativo": round(ruido_immerkaer(gris) / max(float(gris.mean()), 1.0), 3),
        "densidad_bordes_pct": round(densidad_bordes(gris), 1),
    }


def etiquetar_problemas(df: pd.DataFrame, fila_ref: int = 0) -> pd.DataFrame:
    """Compara cada imagen contra la de referencia y asigna etiquetas.
    Los umbrales son relativos a la referencia porque una metrica aislada
    (p. ej. varianza del Laplaciano) depende de la escena y la resolucion."""
    ref = df.iloc[fila_ref]
    etiquetas = []
    for _, r in df.iterrows():
        p = []
        if r.brillo_media < 70 or r.pct_oscuros > 40:
            p.append("subexpuesta")
        if r.pct_saturados > 5:
            p.append("sobreexpuesta")
        if r.contraste_rms < 0.6 * ref.contraste_rms:
            p.append("bajo contraste")
        if r.nitidez_lap_var < 0.35 * ref.nitidez_lap_var and r.brillo_media >= 70:
            p.append("desenfocada")
        # ruido relativo = sigma / brillo (inverso de la relacion senal-ruido)
        if r.ruido_relativo > 2.0 * ref.ruido_relativo:
            p.append("ruido (baja SNR)")
        # en imagenes oscuras Canny marca ruido como borde: se excluyen
        if r.densidad_bordes_pct > 1.3 * ref.densidad_bordes_pct and r.brillo_media >= 70:
            p.append("fondo con exceso de bordes")
        if r.megapixeles < 0.8 * ref.megapixeles:
            p.append("menor resolucion")
        etiquetas.append(", ".join(p) if p else "sin problemas")
    df = df.copy()
    df["problemas"] = etiquetas
    return df


def main(carpeta: str, salida: str) -> pd.DataFrame:
    rutas = sorted(glob.glob(os.path.join(carpeta, "*.*")))
    df = etiquetar_problemas(pd.DataFrame([diagnosticar(r) for r in rutas]))
    os.makedirs(salida, exist_ok=True)
    df.to_csv(os.path.join(salida, "tabla_diagnostico.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="imagenes")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    print(main(a.carpeta, a.salida).T)
