# 🍽️ MobileNet Kitchen Framework

Framework interactivo para entrenar modelos de visión artificial sobre contenido de heladeras. Permite scrapear imágenes, construir un dataset, entrenar una red neuronal y probar predicciones, todo desde una interfaz web local o en VPS.

---

## ¿Para qué sirve?

Dado que querés que un sistema reconozca alimentos dentro de una heladera, este framework te permite:

1. **Scrapear** imágenes de internet por categoría (lechuga, tomate, pollo, etc.)
2. **Inspeccionar** el dataset generado y su distribución
3. **Entrenar** un modelo MobileNet o EfficientNet con tus imágenes
4. **Probar** el modelo con fotos reales de tu heladera

---

## Instalación

### Requisitos

- Python 3.10 o superior
- pip
- ~2 GB de espacio en disco (modelos + dataset)
- GPU opcional (CUDA) — funciona bien en CPU

### Pasos

```bash
# Clonar el repositorio
git clone https://github.com/kos0-CoL/entrenamiento-qch
cd entrenamiento-qch

# Crear entorno virtual
python3 -m venv venv

# Activar el entorno
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Instalar dependencias
pip install gradio torch torchvision duckduckgo-search requests Pillow matplotlib
```

---

## Uso

### Arranque local

```bash
source venv/bin/activate
python mobilenet_kitchen.py
```

Abre el navegador en: [http://localhost:7860](http://localhost:7860)

### Arranque en VPS (acceso público)

```bash
python mobilenet_kitchen.py --share
```

Gradio genera un enlace público temporal (`https://xxxx.gradio.live`).

### Puerto personalizado

```bash
python mobilenet_kitchen.py --port 8080
```

### Script de arranque rápido (opcional)

```bash
chmod +x run.sh
./run.sh           # local
./run.sh --share   # VPS
```

---

## Flujo de trabajo

### 1. 📥 Scraper

- Seleccioná clases de la lista predefinida (verduras, lácteos, carnes, bebidas, contexto de heladera)
- O escribí tus propias clases separadas por coma
- Elegí cuántas imágenes por clase (recomendado: mínimo 30)
- Las imágenes se guardan en `dataset_heladera/<nombre_clase>/`

### 2. 🗂️ Dataset

- Revisá la distribución de imágenes por clase
- Eliminá clases con pocas imágenes o resultados erróneos
- Necesitás al menos **2 clases** con imágenes para entrenar

### 3. 🔥 Entrenar

Elegí el modelo según tu hardware:

| Modelo | Tamaño | Velocidad | Precisión |
|---|---|---|---|
| MobileNetV3-Small | ~10 MB | ⚡⚡⚡ | ★★☆ |
| MobileNetV3-Large | ~22 MB | ⚡⚡☆ | ★★★ |
| MobileNetV2 | ~14 MB | ⚡⚡☆ | ★★☆ |
| EfficientNet-B0 | ~21 MB | ⚡⚡☆ | ★★★ |

Configuración recomendada para empezar:
- Épocas: 5–10
- Learning rate: 0.001
- Batch size: 8
- Data augmentation: ✅ activado
- Freeze backbone: ❌ desactivado (salvo que tengas pocas imágenes)

El modelo se descarga automáticamente la primera vez.

### 4. 🧪 Probar

- Subí una foto desde tu computadora o usá la webcam
- El sistema usa el modelo del **último entrenamiento** automáticamente
- Muestra el top-5 de predicciones con probabilidades

---

## Estructura de carpetas

```
entrenamiento-qch/
├── mobilenet_kitchen.py        ← script principal
├── run.sh                      ← script de arranque rápido
├── README.md
├── dataset_heladera/           ← generado al scrapear
│   ├── lechuga/
│   ├── tomate/
│   └── ...
├── modelos_guardados/          ← generado al entrenar
│   ├── mobilenet_v3_small.pth
│   └── ultimo_entrenamiento.json
└── outputs/                    ← gráficos generados
    ├── dataset_dist.png
    ├── training_plot.png
    └── inferencia_plot.png
```

> Las carpetas `dataset_heladera/`, `modelos_guardados/` y `outputs/` están en `.gitignore` — no se suben al repo.

---

## Dependencias

```
gradio
torch
torchvision
duckduckgo-search
requests
Pillow
matplotlib
```

---

## Notas

- El scraping usa DuckDuckGo Images, no requiere API key
- Los modelos preentrenados se descargan de PyTorch Hub (requiere internet la primera vez)
- Si tenés GPU con CUDA, el entrenamiento la detecta automáticamente
- El archivo `ultimo_entrenamiento.json` guarda las clases y la ruta del modelo para que la pestaña **Probar** funcione sin configuración adicional
