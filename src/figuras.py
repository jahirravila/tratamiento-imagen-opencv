"""
figuras.py
----------
Genera las figuras del informe a partir de las imagenes y las tablas CSV.

Uso:
    python src/figuras.py --imagenes imagenes --resultados resultados
"""
import argparse
import glob
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
# paleta categorica en orden fijo (una por imagen; la referencia en gris neutro)
COLORES = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
NOMBRES = ["Referencia", "Baja iluminación", "Desenfoque/movimiento",
           "Fondo complejo", "Perspectiva"]


def rgb(ruta):
    return cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB)


def conjunto(rutas, salida):
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.3))
    for a, r, n in zip(ax, rutas, NOMBRES):
        a.imshow(rgb(r)); a.set_title(n); a.axis("off")
    fig.savefig(os.path.join(salida, "fig_conjunto.png")); plt.close(fig)


def histogramas(rutas, salida):
    fig, ax = plt.subplots(1, 5, figsize=(12, 1.9), sharey=True)
    for a, r, n, c in zip(ax, rutas, NOMBRES, COLORES):
        g = cv2.cvtColor(cv2.imread(r), cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
        h /= h.sum()
        a.fill_between(range(256), h, color=c, alpha=0.35, lw=0)
        a.plot(h, color=c, lw=1.4)
        a.set_title(n); a.set_xlim(0, 255); a.set_xticks([0, 128, 255])
        a.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax[0].set_ylabel("Frecuencia relativa")
    fig.supxlabel("Nivel de gris", fontsize=9, color="#52514e", y=-0.08)
    fig.savefig(os.path.join(salida, "fig_histogramas.png")); plt.close(fig)


def metricas_indexadas(df, salida):
    """Cada metrica indexada a la referencia (=100) para comparar escalas distintas."""
    cols = [("brillo_media", "Brillo"), ("contraste_rms", "Contraste RMS"),
            ("nitidez_lap_var", "Nitidez (var. Laplaciano)"),
            ("ruido_relativo", "Ruido relativo (sigma/brillo)"),
            ("densidad_bordes_pct", "Densidad de bordes")]
    fig, ax = plt.subplots(1, len(cols), figsize=(12, 2.4), sharey=True)
    y = np.arange(len(df))[::-1]
    for a, (c, t) in zip(ax, cols):
        v = df[c] / df[c].iloc[0] * 100
        a.barh(y, v, color=COLORES, height=0.62, edgecolor="white", lw=2)
        a.axvline(100, color="#52514e", lw=0.8, ls="--")
        for yi, vi in zip(y, v):
            a.text(vi + 4, yi, f"{vi:.0f}", va="center", fontsize=7.5, color="#0b0b0b")
        a.set_title(t); a.set_xlim(0, max(v.max() * 1.25, 130))
        a.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax[0].set_yticks(y); ax[0].set_yticklabels(NOMBRES)
    fig.supxlabel("Índice respecto a la referencia (referencia = 100)", fontsize=9,
                  color="#52514e", y=-0.06)
    fig.savefig(os.path.join(salida, "fig_metricas.png")); plt.close(fig)


def bordes(rutas, salida):
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.3))
    for a, i in zip(ax, [0, 1, 2, 3]):
        g = cv2.cvtColor(cv2.imread(rutas[i]), cv2.COLOR_BGR2GRAY)
        med = np.median(g)
        e = cv2.Canny(g, int(0.66 * med), max(int(1.33 * med), int(0.66 * med) + 1))
        a.imshow(255 - e, cmap="gray"); a.axis("off")
        a.set_title(f"{NOMBRES[i]}  ({(e > 0).mean() * 100:.1f} % bordes)")
    fig.savefig(os.path.join(salida, "fig_bordes.png")); plt.close(fig)


def tecnicas(res, salida):
    items = [("trat_original.png", "Original (baja luz)"), ("trat_a.png", "A. Gamma"),
             ("trat_b.png", "B. Ecualización global"), ("trat_c.png", "C. CLAHE"),
             ("trat_e.png", "E. NL-Means antes de gamma"), ("trat_d.png", "D. Propuesto")]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 4.8))
    for a, (f, t) in zip(ax.ravel(), items):
        a.imshow(rgb(os.path.join(res, f))); a.set_title(t); a.axis("off")
    fig.savefig(os.path.join(salida, "fig_tecnicas.png")); plt.close(fig)


def antes_despues(img_dir, res, salida):
    orig = rgb(os.path.join(img_dir, "2_baja_iluminacion.jpg"))
    trat = rgb(os.path.join(res, "trat_d.png"))
    ref = rgb(os.path.join(img_dir, "1_referencia.jpg"))
    y0, y1, x0, x1 = 200, 330, 300, 430  # recorte: cuchara (zona de detalle fino)
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6),
                           gridspec_kw={"height_ratios": [1, 0.9]})
    for j, (im, t) in enumerate([(orig, "Antes (baja iluminación)"),
                                 (trat, "Después (tratamiento D)"),
                                 (ref, "Referencia")]):
        ax[0, j].imshow(im); ax[0, j].set_title(t); ax[0, j].axis("off")
        ax[0, j].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                         ec="#eda100", lw=1.5))
        ax[1, j].imshow(im[y0:y1, x0:x1]); ax[1, j].axis("off")
        ax[1, j].set_title("Detalle ampliado", fontsize=8.5, color="#52514e")
    fig.savefig(os.path.join(salida, "fig_antes_despues.png")); plt.close(fig)

    fig, a = plt.subplots(figsize=(7, 2.4))
    for im, n, c in [(orig, "Antes", "#2a78d6"), (trat, "Después (D)", "#eb6834"),
                     (ref, "Referencia", "#8a8984")]:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel(); h /= h.sum()
        a.plot(h, color=c, lw=2, label=n)
    a.set_yscale("log"); a.set_ylim(1e-5, 0.2); a.set_xlim(0, 255)
    a.set_xlabel("Nivel de gris"); a.set_ylabel("Frecuencia (log)")
    a.grid(color="#e6e5e0", lw=0.6); a.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig(os.path.join(salida, "fig_hist_antes_despues.png")); plt.close(fig)


def pasos(img_dir, salida):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tratamiento import d_propuesto
    r = d_propuesto(cv2.imread(os.path.join(img_dir, "2_baja_iluminacion.jpg")), pasos=True)
    items = [("1_gamma", "1. Gamma en L"), ("2_nlmeans", "2. NL-Means"),
             ("3_croma", f"3. Croma (k = {r['k_croma']})"), ("4_clahe", "4. CLAHE (clip 2)")]
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.3))
    for a, (k, t) in zip(ax, items):
        a.imshow(cv2.cvtColor(r[k], cv2.COLOR_BGR2RGB)); a.set_title(t); a.axis("off")
    fig.savefig(os.path.join(salida, "fig_pasos.png")); plt.close(fig)


def main(img_dir, res):
    rutas = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    df = pd.read_csv(os.path.join(res, "tabla_diagnostico.csv"))
    conjunto(rutas, res); histogramas(rutas, res); metricas_indexadas(df, res)
    bordes(rutas, res); tecnicas(res, res); pasos(img_dir, res); antes_despues(img_dir, res, res)
    print("Figuras guardadas en", os.path.abspath(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenes", default="imagenes")
    ap.add_argument("--resultados", default="resultados")
    a = ap.parse_args()
    main(a.imagenes, a.resultados)
