"""
tratamiento.py
--------------
Tratamiento de la imagen con BAJA ILUMINACION (problema seleccionado).

Diagnostico que motiva el tratamiento:
  * brillo medio muy bajo y >80 % de pixeles oscuros  -> subexposicion
  * contraste RMS y rango dinamico reducidos           -> histograma comprimido
  * ruido relativo (sigma/brillo) ~4x la referencia    -> baja relacion senal-ruido

Por eso no basta con "aclarar": cualquier realce de brillo amplifica el ruido.
Se comparan tres tecnicas clasicas y una cadena propuesta:
  A. Correccion gamma automatica (solo brillo)
  B. Ecualizacion global del histograma en el canal de luminancia
  C. CLAHE en el canal L de LAB (contraste local, sin tocar el color)
  D. Propuesta: gamma en L -> eliminacion de ruido NL-Means -> CLAHE suave

Uso:
    python src/tratamiento.py --entrada imagenes/2_baja_iluminacion.jpg \
        --referencia imagenes/1_referencia.jpg --salida resultados
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diagnostico import (densidad_bordes, entropia, nitidez_laplaciano,
                         ruido_immerkaer)


def gamma_automatica(canal: np.ndarray, objetivo: float = 0.45) -> np.ndarray:
    """Elige gamma para que la media normalizada pase a `objetivo`:
    media**g = objetivo  =>  g = log(objetivo) / log(media)."""
    media = max(canal.mean() / 255.0, 1e-3)
    g = np.log(objetivo) / np.log(media)
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(canal, lut)


def a_gamma(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = gamma_automatica(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def b_ecualizacion_global(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def c_clahe(img, clip: float = 3.0, rejilla: int = 8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(rejilla, rejilla))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def recuperar_croma(img, k: float):
    """Escala los canales a*, b* de LAB alrededor del neutro (128).
    Al oscurecer una escena el croma cae junto con la luminancia; la gamma
    en L no lo devuelve y el resultado queda 'lavado'."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] = np.clip((lab[..., 1:] - 128) * k + 128, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def d_propuesto(img, h_luz: int = 8, h_color: int = 10, clip: float = 2.0,
                k_max: float = 3.0, pasos: bool = False):
    """1) Gamma en L: recupera los medios tonos sin saturar las luces.
    2) NL-Means: el ruido ya fue amplificado por la gamma y ahora es visible,
       asi que el filtro lo estima mejor; promedia parches similares y conserva
       bordes (a diferencia de un desenfoque gaussiano).
    3) Recuperacion de croma con k = sqrt(ganancia de L), limitado a k_max:
       compromiso entre no restaurar color y restaurarlo proporcionalmente
       (lo que amplificaria tambien el ruido de color).
    4) CLAHE suave: devuelve contraste local limitado (clip bajo) para no
       volver a amplificar el ruido residual."""
    paso1 = a_gamma(img)
    paso2 = cv2.fastNlMeansDenoisingColored(paso1, None, h_luz, h_color, 7, 21)
    L_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean()
    L_out = cv2.cvtColor(paso1, cv2.COLOR_BGR2LAB)[..., 0].mean()
    k = float(min(np.sqrt(L_out / max(L_in, 1.0)), k_max))
    paso3 = recuperar_croma(paso2, k)
    paso4 = c_clahe(paso3, clip=clip, rejilla=8)
    if pasos:
        return {"1_gamma": paso1, "2_nlmeans": paso2, "3_croma": paso3,
                "4_clahe": paso4, "k_croma": round(k, 2)}
    return paso4


def d_orden_invertido(img):
    """Ablacion: mismo filtro pero eliminando ruido ANTES de aclarar.
    Sirve para justificar el orden de la cadena propuesta."""
    p = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 21)
    return c_clahe(a_gamma(p), clip=2.0)  # sin recuperar croma, como A-C


def croma_medio(img) -> float:
    """Saturacion perceptual: media de sqrt(a*^2 + b*^2) en LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2).mean())


def metricas(img, ref=None) -> dict:
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(gris, [2, 98])
    m = {
        "brillo_media": round(float(gris.mean()), 1),
        "contraste_rms": round(float(gris.std()), 1),
        "rango_dinamico_p2_p98": int(p98 - p2),
        "entropia_bits": round(entropia(gris), 2),
        "nitidez_lap_var": round(nitidez_laplaciano(gris), 1),
        "ruido_relativo": round(float(ruido_immerkaer(gris) / max(gris.mean(), 1.0)), 3),
        "densidad_bordes_pct": round(densidad_bordes(gris), 1),
        "croma_medio": round(croma_medio(img), 1),
    }
    if ref is not None and not np.array_equal(ref, img):  # metricas con referencia (posibles porque las vistas estan alineadas)
        m["PSNR_dB"] = round(float(peak_signal_noise_ratio(ref, img)), 2)
        m["SSIM"] = round(float(structural_similarity(ref, img, channel_axis=2)), 3)
    return m


TECNICAS = {
    "Original (baja luz)": lambda x: x,
    "A. Gamma automatica": a_gamma,
    "B. Ecualizacion global": b_ecualizacion_global,
    "C. CLAHE (L, clip 3)": c_clahe,
    "D. Propuesto: gamma+NLM+croma+CLAHE": d_propuesto,
    "E. Ablacion: NLM antes de gamma": d_orden_invertido,
}


def main(entrada, referencia, salida) -> pd.DataFrame:
    img, ref = cv2.imread(entrada), cv2.imread(referencia)
    if ref.shape != img.shape:  # fotos reales: no se pueden usar PSNR/SSIM
        ref = None
    os.makedirs(salida, exist_ok=True)
    filas = []
    for nombre, f in TECNICAS.items():
        out = f(img)
        cv2.imwrite(os.path.join(salida, f"trat_{nombre.split('.')[0].split(' ')[0].lower()}.png"), out)
        filas.append({"tecnica": nombre, **metricas(out, ref)})
    if ref is not None:
        filas.append({"tecnica": "Referencia", **metricas(ref, ref)})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, "tabla_tratamiento.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="imagenes/2_baja_iluminacion.jpg")
    ap.add_argument("--referencia", default="imagenes/1_referencia.jpg")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    print(main(a.entrada, a.referencia, a.salida).to_string(index=False))
