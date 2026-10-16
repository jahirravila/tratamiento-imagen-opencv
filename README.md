# Tratamiento de imagen con OpenCV

Actividad de la Semana 8 — Especialización en Inteligencia Artificial (Universidad de Cundinamarca).
Diagnóstico de calidad de 5 imágenes de un mismo objeto capturadas en condiciones distintas y tratamiento
justificado de la imagen problemática (baja iluminación).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1plRW2NRQ-C6CmlRpb8NcssRJVmaTBRVx?usp=sharing)

## Estructura
| Ruta | Contenido |
|---|---|
| `notebooks/tratamiento_imagen.ipynb` | Notebook principal (Colab), ejecutado con resultados |
| `src/generar_imagenes_prueba.py` | Genera el conjunto de prueba simulado (5 condiciones) |
| `src/diagnostico.py` | Métricas: resolución, canales, brillo, contraste, nitidez, ruido, bordes, formato y tamaño |
| `src/tratamiento.py` | Técnicas A–E para baja iluminación y métricas antes/después (incl. PSNR y SSIM) |
| `src/figuras.py` | Figuras del informe |
| `imagenes/` | Las 5 imágenes analizadas |
| `resultados/` | Tablas CSV y figuras |

## Uso local
```bash
pip install -r requirements.txt
python src/generar_imagenes_prueba.py   # solo si no tiene fotos propias
python src/diagnostico.py
python src/tratamiento.py
python src/figuras.py
```

## Usar fotos propias
Reemplace los archivos de `imagenes/` conservando los prefijos
`1_referencia`, `2_baja_iluminacion`, `3_desenfoque_movimiento`, `4_fondo_complejo`, `5_perspectiva`
(.jpg o .png) y vuelva a ejecutar el notebook. Si las fotos no están alineadas con la referencia,
PSNR/SSIM se omiten automáticamente y la evaluación usa solo métricas sin referencia.
