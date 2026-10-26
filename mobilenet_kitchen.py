"""
╔══════════════════════════════════════════════════════════╗
║        MobileNet Kitchen Framework  v2.0                 ║
║  Scraper  →  Dataset  →  Entrenar  →  Probar             ║
╚══════════════════════════════════════════════════════════╝

Dependencias:
    pip install gradio torch torchvision duckduckgo-search requests Pillow matplotlib

Uso local:
    python mobilenet_kitchen.py

Uso en VPS (acceso público):
    python mobilenet_kitchen.py --share
"""

import os
import sys
import time
import json
import shutil
import argparse
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from ddgs import DDGS

# ─────────────────────────────────────────────
#  CONFIGURACIÓN CENTRAL DEL FRAMEWORK
# ─────────────────────────────────────────────

DATASET_ROOT = "./dataset_heladera"
MODELS_DIR   = "./modelos_guardados"
OUTPUTS_DIR  = "./outputs"

os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(OUTPUTS_DIR,  exist_ok=True)

# Lista predefinida de clases para heladera
CLASES_PREDEFINIDAS = [
    # Verduras y frutas
    "lechuga",
    "tomate",
    "zanahoria",
    "pimiento rojo",
    "pimiento verde",
    "cebolla",
    "manzana",
    "naranja",
    "limón",
    "uvas",
    # Lácteos
    "leche",
    "yogur",
    "manteca",
    "crema",
    "queso",
    # Carnes y proteínas
    "pollo crudo",
    "carne vacuna",
    "huevos",
    "fiambres",
    # Bebidas
    "jugo tetrabrik",
    "agua mineral",
    "cerveza lata",
    # Condimentos / otros
    "mayonesa",
    "ketchup",
    "mostaza",
    # Contexto de heladera
    "heladera abierta con alimentos",
    "estante de heladera lleno",
    "heladera vacía",
]

# Catálogo de modelos disponibles
CATALOGO_MODELOS = {
    "MobileNetV3-Small  ⚡ Ultra liviano (~10 MB)": {
        "fn":        lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "clf_attr":  "classifier",
        "in_feat_fn": lambda m: m.classifier[-1].in_features,
        "filename":  "mobilenet_v3_small.pth",
    },
    "MobileNetV3-Large  🎯 Más preciso (~22 MB)": {
        "fn":        lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
        "clf_attr":  "classifier",
        "in_feat_fn": lambda m: m.classifier[-1].in_features,
        "filename":  "mobilenet_v3_large.pth",
    },
    "MobileNetV2       ⚖️ Equilibrado (~14 MB)": {
        "fn":        lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        "clf_attr":  "classifier",
        "in_feat_fn": lambda m: m.classifier[-1].in_features,
        "filename":  "mobilenet_v2.pth",
    },
    "EfficientNet-B0   🚀 Alta eficiencia (~21 MB)": {
        "fn":        lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "clf_attr":  "classifier",
        "in_feat_fn": lambda m: m.classifier[-1].in_features,
        "filename":  "efficientnet_b0.pth",
    },
}

# Metadatos del entrenamiento actual (para que "Probar" sepa qué modelo usar)
METADATA_PATH = os.path.join(MODELS_DIR, "ultimo_entrenamiento.json")


# ─────────────────────────────────────────────
#  HELPERS COMPARTIDOS
# ─────────────────────────────────────────────

def listar_clases_en_dataset(root=DATASET_ROOT):
    """Devuelve dict {clase: cantidad_imagenes} para todas las subcarpetas del dataset."""
    if not os.path.exists(root):
        return {}
    clases = {}
    for nombre in sorted(os.listdir(root)):
        carpeta = os.path.join(root, nombre)
        if os.path.isdir(carpeta):
            imgs = [f for f in os.listdir(carpeta)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            clases[nombre] = len(imgs)
    return clases


def guardar_metadata(modelo_key, clases, archivo_modelo):
    data = {
        "modelo_key":    modelo_key,
        "clases":        clases,
        "archivo_modelo": archivo_modelo,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cargar_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  TAB 1 – SCRAPER
# ─────────────────────────────────────────────

def scrape_imagenes(
    clases_seleccionadas,
    clase_personalizada,
    cantidad,
    progress=gr.Progress()
):
    """Descarga imágenes para cada clase usando DuckDuckGo Images."""

    # Unir clases seleccionadas + clase personalizada (si la escribió)
    clases = list(clases_seleccionadas) if clases_seleccionadas else []
    extra = [c.strip() for c in clase_personalizada.split(",") if c.strip()]
    clases += extra

    if not clases:
        return "⚠️ Seleccioná al menos una clase o escribí una personalizada.", None

    cantidad = max(1, int(cantidad))
    log_lines = []
    previews  = []

    for idx, clase in enumerate(clases):
        nombre_carpeta = clase.replace(" ", "_").lower()
        ruta = os.path.join(DATASET_ROOT, nombre_carpeta)

        progress(idx / len(clases), desc=f"🔍 Buscando '{clase}'…")

        try:
            ddgs = DDGS()
            resultados = list(ddgs.images(clase, max_results=cantidad))
            urls = [r["image"] for r in resultados if "image" in r]
        except Exception as e:
            log_lines.append(f"❌ Error buscando '{clase}': {e}")
            continue

        descargadas = 0
        for i, url in enumerate(urls):
            try:
                r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    ext = url.split(".")[-1].split("?")[0].lower()
                    if ext not in ("jpg", "jpeg", "png"):
                        ext = "jpg"
                    os.makedirs(ruta, exist_ok=True)
                    fname = os.path.join(ruta, f"img_{int(time.time()*1000)}_{i}.{ext}")
                    with open(fname, "wb") as f:
                        f.write(r.content)
                    descargadas += 1
            except Exception:
                continue

        log_lines.append(f"✅ '{clase}' → {descargadas} imgs en '{nombre_carpeta}/'")

        # Tomar hasta 2 previews por clase
        imgs_en_carpeta = [
            os.path.join(ruta, f)
            for f in (os.listdir(ruta) if os.path.exists(ruta) else [])
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:2]
        previews.extend(imgs_en_carpeta)

    log_lines.append("")
    log_lines.append(f"📁 Dataset root: {os.path.abspath(DATASET_ROOT)}")
    return "\n".join(log_lines), previews[:8] if previews else None


# ─────────────────────────────────────────────
#  TAB 2 – DATASET INSPECTOR
# ─────────────────────────────────────────────

def refrescar_dataset():
    clases = listar_clases_en_dataset()
    if not clases:
        return (
            "📭 El dataset está vacío. Usá la pestaña **Scraper** para descargar imágenes.",
            None
        )

    total = sum(clases.values())
    lines = [f"📂 Dataset: `{os.path.abspath(DATASET_ROOT)}`",
             f"🗂️ Clases: {len(clases)}   |   🖼️ Total imágenes: {total}", ""]

    for clase, n in sorted(clases.items(), key=lambda x: -x[1]):
        barra = "█" * min(n // 5, 40)
        lines.append(f"  {clase:<30} {n:>4} imgs  {barra}")

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(max(6, len(clases) * 0.5), 4))
    nombres = list(clases.keys())
    cantidades = [clases[n] for n in nombres]
    ax.barh(nombres, cantidades, color="#5b7fa6")
    ax.set_xlabel("Imágenes")
    ax.set_title("Distribución del dataset")
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, "dataset_dist.png")
    plt.savefig(plot_path)
    plt.close(fig)

    return "\n".join(lines), plot_path


def eliminar_clase(clase_a_borrar):
    if not clase_a_borrar.strip():
        return "⚠️ Escribí el nombre exacto de la subcarpeta a eliminar."
    ruta = os.path.join(DATASET_ROOT, clase_a_borrar.strip())
    if not os.path.exists(ruta):
        return f"❌ No existe la carpeta '{ruta}'."
    shutil.rmtree(ruta)
    return f"🗑️ Eliminada la clase '{clase_a_borrar}' del dataset."


# ─────────────────────────────────────────────
#  TAB 3 – ENTRENAMIENTO
# ─────────────────────────────────────────────

def entrenar_modelo(
    modelo_key,
    epochs,
    lr,
    batch_size,
    augment,
    freeze_backbone,
    progress=gr.Progress()
):
    clases_dict = listar_clases_en_dataset()
    if len(clases_dict) < 2:
        return (
            "⚠️ Necesitás al menos 2 clases con imágenes en el dataset para entrenar.",
            None
        )

    cfg = CATALOGO_MODELOS[modelo_key]
    epochs     = int(epochs)
    batch_size = int(batch_size)
    lr         = float(lr)

    # ── Transforms ──────────────────────────────────────
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── Limpiar imágenes corruptas antes de entrenar ────────
    progress(0.01, desc="🧹 Verificando imágenes del dataset…")
    borradas = 0
    for clase in os.listdir(DATASET_ROOT):
        carpeta = os.path.join(DATASET_ROOT, clase)
        if not os.path.isdir(carpeta):
            continue
        for fname in os.listdir(carpeta):
            fpath = os.path.join(carpeta, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                os.remove(fpath)
                borradas += 1
    if borradas:
        progress(0.02, desc=f"🧹 {borradas} imágenes corruptas eliminadas")

    progress(0.02, desc="📦 Cargando imágenes…")
    try:
        dataset    = datasets.ImageFolder(root=DATASET_ROOT, transform=train_tf)
        num_clases = len(dataset.classes)
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
    except Exception as e:
        return f"⚠️ Error cargando el dataset: {e}", None

    # ── Modelo ───────────────────────────────────────────
    progress(0.05, desc=f"⬇️ Descargando / cargando {modelo_key}…")
    try:
        model = cfg["fn"]()
    except Exception as e:
        return f"⚠️ Error al cargar el modelo: {e}", None

    # Reemplazar la última capa de clasificación
    in_feat = cfg["in_feat_fn"](model)
    clf = getattr(model, cfg["clf_attr"])

    if isinstance(clf, nn.Sequential):
        clf[-1] = nn.Linear(in_feat, num_clases)
    else:
        setattr(model, cfg["clf_attr"], nn.Linear(in_feat, num_clases))

    # Congelar backbone si se eligió fine-tuning ligero
    if freeze_backbone:
        for name, param in model.named_parameters():
            if cfg["clf_attr"] not in name:
                param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    # ── Loop de entrenamiento ───────────────────────────
    losses    = []
    accs      = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc  = 100.0 * correct / total
        losses.append(epoch_loss)
        accs.append(epoch_acc)
        scheduler.step()

        progress(
            (epoch + 1) / epochs,
            desc=f"🔄 Época {epoch+1}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.1f}%"
        )

    # ── Guardar modelo ───────────────────────────────────
    archivo_modelo = os.path.join(MODELS_DIR, cfg["filename"])
    torch.save(model.state_dict(), archivo_modelo)
    guardar_metadata(modelo_key, dataset.classes, archivo_modelo)

    # ── Gráfico ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, epochs+1), losses, "o-", color="#d44", label="Loss")
    ax1.set_title("Loss por época"); ax1.set_xlabel("Época"); ax1.grid(True)
    ax2.plot(range(1, epochs+1), accs,  "o-", color="#48a",  label="Accuracy %")
    ax2.set_title("Accuracy por época"); ax2.set_xlabel("Época"); ax2.set_ylim(0, 100); ax2.grid(True)
    plt.suptitle(f"Entrenamiento – {modelo_key.split()[0]}", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, "training_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)

    msg = (
        f"✅ Entrenamiento completado en **{device.type.upper()}**\n\n"
        f"Modelo:    {modelo_key}\n"
        f"Épocas:    {epochs}\n"
        f"Clases ({num_clases}):  {', '.join(dataset.classes)}\n"
        f"Loss final:  {losses[-1]:.4f}\n"
        f"Acc final:   {accs[-1]:.1f}%\n\n"
        f"💾 Guardado en: {archivo_modelo}"
    )
    return msg, plot_path


# ─────────────────────────────────────────────
#  TAB 4 – PROBAR MODELO
# ─────────────────────────────────────────────

def inferir(imagen_path):
    if imagen_path is None:
        return "⚠️ No subiste ninguna imagen.", None

    meta = cargar_metadata()
    if meta is None:
        return "⚠️ No hay modelo entrenado aún. Primero completá el entrenamiento.", None

    modelo_key    = meta["modelo_key"]
    clases        = meta["clases"]
    archivo_modelo = meta["archivo_modelo"]

    if not os.path.exists(archivo_modelo):
        return f"⚠️ No se encontró el archivo del modelo: {archivo_modelo}", None

    cfg       = CATALOGO_MODELOS[modelo_key]
    num_clases = len(clases)

    # Reconstruir arquitectura
    model = cfg["fn"]()
    in_feat = cfg["in_feat_fn"](model)
    clf     = getattr(model, cfg["clf_attr"])
    if isinstance(clf, nn.Sequential):
        clf[-1] = nn.Linear(in_feat, num_clases)
    else:
        setattr(model, cfg["clf_attr"], nn.Linear(in_feat, num_clases))

    model.load_state_dict(torch.load(archivo_modelo, map_location="cpu"))
    model.eval()

    # Preprocesamiento
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(imagen_path).convert("RGB")
    tensor = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_k = min(5, num_clases)
    top_probs, top_idx = probs.topk(top_k)
    top_probs = top_probs.numpy()
    top_idx   = top_idx.numpy()

    # Resultado textual
    lines = ["🔍 **Resultados de la inferencia:**\n"]
    for rank, (prob, idx) in enumerate(zip(top_probs, top_idx)):
        clase = clases[idx]
        bar   = "█" * int(prob * 30)
        mark  = " ⬅️ PREDICCIÓN" if rank == 0 else ""
        lines.append(f"  {rank+1}. {clase:<28} {prob*100:5.1f}%  {bar}{mark}")

    # Gráfico de barras horizontales
    fig, ax = plt.subplots(figsize=(7, 3))
    nombres_top = [clases[i] for i in top_idx]
    ax.barh(nombres_top[::-1], top_probs[::-1] * 100, color="#5b7fa6")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilidad (%)")
    ax.set_title("Top predicciones")
    plt.tight_layout()
    result_plot = os.path.join(OUTPUTS_DIR, "inferencia_plot.png")
    plt.savefig(result_plot)
    plt.close(fig)

    return "\n".join(lines), result_plot



# ─────────────────────────────────────────────
#  TAB 5 – DETECCIÓN POR SLIDING WINDOW
# ─────────────────────────────────────────────

def detectar_en_heladera(imagen_path, filas, columnas, umbral_confianza, overlap):
    if imagen_path is None:
        return "⚠️ No subiste ninguna imagen.", None

    meta = cargar_metadata()
    if meta is None:
        return "⚠️ No hay modelo entrenado aún.", None

    modelo_key     = meta["modelo_key"]
    clases         = meta["clases"]
    archivo_modelo = meta["archivo_modelo"]

    if not os.path.exists(archivo_modelo):
        return f"⚠️ No se encontró el modelo: {archivo_modelo}", None

    cfg        = CATALOGO_MODELOS[modelo_key]
    num_clases = len(clases)

    model   = cfg["fn"]()
    in_feat = cfg["in_feat_fn"](model)
    clf     = getattr(model, cfg["clf_attr"])
    if isinstance(clf, nn.Sequential):
        clf[-1] = nn.Linear(in_feat, num_clases)
    else:
        setattr(model, cfg["clf_attr"], nn.Linear(in_feat, num_clases))

    model.load_state_dict(torch.load(archivo_modelo, map_location="cpu"))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    import hashlib
    from PIL import ImageDraw

    def color_clase(nombre):
        h = int(hashlib.md5(nombre.encode()).hexdigest()[:6], 16)
        return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)

    IGNORAR = {"vacio", "vacia", "vacío", "vacía", "fondo", "background", "heladera_abierta_con_alimentos", "heladera abierta con alimentos", "estante_de_heladera_lleno", "heladera_vacía", "heladera vacía"}

    img_original = Image.open(imagen_path).convert("RGB")
    ancho, alto  = img_original.size
    filas        = int(filas)
    columnas     = int(columnas)
    overlap      = float(overlap)

    # Tamaño de celda base
    ancho_celda = ancho // columnas
    alto_celda  = alto  // filas

    # Paso con overlap: paso = celda * (1 - overlap)
    paso_x = max(1, int(ancho_celda * (1 - overlap)))
    paso_y = max(1, int(alto_celda  * (1 - overlap)))

    # Generar todas las ventanas deslizantes
    ventanas = []  # (x1, y1, x2, y2)
    y = 0
    while y + alto_celda <= alto:
        x = 0
        while x + ancho_celda <= ancho:
            ventanas.append((x, y, x + ancho_celda, y + alto_celda))
            x += paso_x
        y += paso_y

    # Clasificar cada ventana
    detecciones_raw = []  # (x1, y1, x2, y2, clase, prob)

    for (x1, y1, x2, y2) in ventanas:
        celda  = img_original.crop((x1, y1, x2, y2))
        tensor = tf(celda).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        prob_max, idx_max = probs.max(0)
        prob_max  = prob_max.item()
        clase_det = clases[idx_max.item()]

        if prob_max < umbral_confianza:
            continue
        if clase_det.lower() in IGNORAR:
            continue

        detecciones_raw.append((x1, y1, x2, y2, clase_det, prob_max))

    # ── Fusión de cajas solapadas (NMS simple por clase) ──
    def iou(a, b):
        ax1,ay1,ax2,ay2 = a[:4]
        bx1,by1,bx2,by2 = b[:4]
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    # Ordenar por confianza descendente
    detecciones_raw.sort(key=lambda d: -d[5])
    fusionadas = []
    usadas = [False] * len(detecciones_raw)

    for i, det in enumerate(detecciones_raw):
        if usadas[i]:
            continue
        grupo = [det]
        usadas[i] = True
        for j, det2 in enumerate(detecciones_raw):
            if usadas[j]:
                continue
            if det2[4] == det[4] and iou(det, det2) > 0.3:
                grupo.append(det2)
                usadas[j] = True
        # Caja fusionada = promedio ponderado por confianza
        total_prob = sum(d[5] for d in grupo)
        fx1 = int(sum(d[0]*d[5] for d in grupo) / total_prob)
        fy1 = int(sum(d[1]*d[5] for d in grupo) / total_prob)
        fx2 = int(sum(d[2]*d[5] for d in grupo) / total_prob)
        fy2 = int(sum(d[3]*d[5] for d in grupo) / total_prob)
        fusionadas.append((fx1, fy1, fx2, fy2, det[4], max(d[5] for d in grupo)))

    # ── Dibujar resultado ────────────────────────────────
    img_draw = img_original.copy()
    draw     = ImageDraw.Draw(img_draw)
    detecciones_finales = {}

    for (x1, y1, x2, y2, clase, prob) in fusionadas:
        color = color_clase(clase)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label    = f"{clase} {prob*100:.0f}%"
        bbox_txt = draw.textbbox((x1 + 4, y1 + 4), label)
        draw.rectangle(bbox_txt, fill=color)
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255))
        detecciones_finales[clase] = detecciones_finales.get(clase, 0) + 1

    result_path = os.path.join(OUTPUTS_DIR, "sliding_window_result.jpg")
    img_draw.save(result_path)

    if detecciones_finales:
        lines = ["🔍 **Alimentos detectados en la heladera:**\n"]
        for clase, n in sorted(detecciones_finales.items(), key=lambda x: -x[1]):
            lines.append(f"  • {clase:<28} x{n}")
        lines.append(f"\n📐 Grilla: {filas}×{columnas}  Overlap: {int(overlap*100)}%  Umbral: {int(umbral_confianza*100)}%")
        lines.append(f"🪟 Ventanas evaluadas: {len(ventanas)}  →  Detecciones fusionadas: {len(fusionadas)}")
    else:
        lines = [
            "🤔 No se detectó nada con suficiente confianza.\n",
            "Sugerencias:",
            "  • Bajá el umbral de confianza",
            "  • Subí el overlap (más solapamiento)",
            "  • Re-entrenás con más imágenes por clase",
        ]

    return "\n".join(lines), result_path


# ─────────────────────────────────────────────
#  INTERFAZ GRADIO
# ─────────────────────────────────────────────

CUSTOM_CSS = """
#main-title { text-align: center; padding: 16px 0 4px; }
#main-title h1 { font-size: 1.6rem; font-weight: 700; }
.tab-content { padding: 12px 0; }
.log-box textarea { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.82rem; }
.status-ok  { color: #3a8a3a; font-weight: 600; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="MobileNet Kitchen") as demo:

    # ── Encabezado ──────────────────────────────────────
    with gr.Row(elem_id="main-title"):
        gr.Markdown(
            "# 🍽️ MobileNet Kitchen Framework\n"
            "Scraper · Dataset · Entrenar · Probar · Deployment-ready"
        )

    with gr.Tabs():

        # ══════════════════════════════════════════════════
        #  TAB 1 — SCRAPER
        # ══════════════════════════════════════════════════
        with gr.TabItem("📥  Scraper"):
            gr.Markdown(
                "Descargá imágenes automáticamente para construir tu dataset. "
                "Seleccioná clases de la lista o escribí las tuyas separadas por coma."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    scr_clases   = gr.CheckboxGroup(
                        choices=CLASES_PREDEFINIDAS,
                        label="Clases predefinidas (heladera)",
                        value=["lechuga", "tomate", "pollo crudo", "heladera abierta con alimentos"],
                    )
                    scr_custom   = gr.Textbox(
                        label="Clases personalizadas (separadas por coma)",
                        placeholder="Ej: budín de pan, frasco de mermelada"
                    )
                    scr_cantidad = gr.Slider(5, 100, value=20, step=5, label="Imágenes por clase")
                    scr_btn      = gr.Button("🚀 Iniciar descarga", variant="primary")

                with gr.Column(scale=1):
                    scr_log     = gr.Textbox(
                        label="Log del scraper", lines=14,
                        interactive=False, elem_classes="log-box"
                    )
                    scr_gallery = gr.Gallery(
                        label="Vista previa (hasta 8 imgs)", columns=4, height=220
                    )

            scr_btn.click(
                fn=scrape_imagenes,
                inputs=[scr_clases, scr_custom, scr_cantidad],
                outputs=[scr_log, scr_gallery],
            )

        # ══════════════════════════════════════════════════
        #  TAB 2 — DATASET
        # ══════════════════════════════════════════════════
        with gr.TabItem("🗂️  Dataset"):
            gr.Markdown("Inspeccioná y gestioná las clases del dataset antes de entrenar.")
            with gr.Row():
                ds_refresh_btn = gr.Button("🔄 Refrescar", variant="secondary")
                ds_del_input   = gr.Textbox(
                    label="Nombre exacto de subcarpeta a eliminar",
                    placeholder="Ej: lechuga"
                )
                ds_del_btn = gr.Button("🗑️ Eliminar clase", variant="stop")

            with gr.Row():
                ds_log  = gr.Textbox(label="Estado del dataset", lines=18,
                                     interactive=False, elem_classes="log-box")
                ds_plot = gr.Image(label="Distribución de clases", height=320)

            ds_refresh_btn.click(fn=refrescar_dataset, outputs=[ds_log, ds_plot])
            ds_del_btn.click(
                fn=eliminar_clase,
                inputs=[ds_del_input],
                outputs=[ds_log],
            )

        # ══════════════════════════════════════════════════
        #  TAB 3 — ENTRENAMIENTO
        # ══════════════════════════════════════════════════
        with gr.TabItem("🔥  Entrenar"):
            gr.Markdown(
                "Elegí el modelo, configurá los hiperparámetros y arrancá el entrenamiento. "
                "El modelo se descarga automáticamente si no está en caché."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    tr_modelo  = gr.Dropdown(
                        choices=list(CATALOGO_MODELOS.keys()),
                        value=list(CATALOGO_MODELOS.keys())[0],
                        label="Arquitectura del modelo"
                    )
                    with gr.Row():
                        tr_epochs = gr.Slider(1, 50, value=5,  step=1,  label="Épocas")
                        tr_lr     = gr.Dropdown(
                            choices=["0.0001", "0.0003", "0.001", "0.003", "0.01"],
                            value="0.001",
                            label="Learning rate"
                        )
                        tr_batch  = gr.Slider(4, 64, value=8, step=4, label="Batch size")

                    with gr.Row():
                        tr_augment  = gr.Checkbox(value=True,  label="Data augmentation")
                        tr_freeze   = gr.Checkbox(value=False, label="Freeze backbone (solo fine-tune)")

                    tr_btn = gr.Button("⚡ Comenzar entrenamiento", variant="primary", size="lg")

                with gr.Column(scale=1):
                    tr_log  = gr.Textbox(label="Estado del entrenamiento", lines=12,
                                         interactive=False, elem_classes="log-box")
                    tr_plot = gr.Image(label="Loss & Accuracy", height=300)

            tr_btn.click(
                fn=entrenar_modelo,
                inputs=[tr_modelo, tr_epochs, tr_lr, tr_batch, tr_augment, tr_freeze],
                outputs=[tr_log, tr_plot],
            )

        # ══════════════════════════════════════════════════
        #  TAB 4 — PROBAR (clasificador simple)
        # ══════════════════════════════════════════════════
        with gr.TabItem("🧪  Probar"):
            gr.Markdown(
                "Subí una foto (de una heladera o de un alimento) y el modelo te dice "
                "qué ve. Usa el modelo del último entrenamiento."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inf_img = gr.Image(
                        type="filepath", label="Subí tu foto aquí",
                        sources=["upload", "webcam"], height=300
                    )
                    inf_btn = gr.Button("🔍 Analizar imagen", variant="primary")

                with gr.Column(scale=1):
                    inf_log  = gr.Textbox(label="Resultado", lines=10,
                                          interactive=False, elem_classes="log-box")
                    inf_plot = gr.Image(label="Distribución de probabilidades", height=240)

            inf_btn.click(
                fn=inferir,
                inputs=[inf_img],
                outputs=[inf_log, inf_plot],
            )

        # ══════════════════════════════════════════════════
        #  TAB 5 — DETECCIÓN EN HELADERA (sliding window)
        # ══════════════════════════════════════════════════
        with gr.TabItem("🧊  Detectar en Heladera"):
            gr.Markdown(
                "Subí una foto de tu heladera abierta. "
                "El sistema la divide en celdas y clasifica cada una por separado, "
                "mostrando qué alimentos encontró y dónde."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sw_img    = gr.Image(
                        type="filepath", label="Foto de la heladera",
                        sources=["upload", "webcam"], height=300
                    )
                    with gr.Row():
                        sw_filas = gr.Slider(2, 8, value=3, step=1, label="Filas de la grilla")
                        sw_cols  = gr.Slider(2, 8, value=3, step=1, label="Columnas de la grilla")
                    sw_umbral  = gr.Slider(0.3, 0.99, value=0.6, step=0.05,
                                           label="Umbral de confianza")
                    sw_overlap = gr.Slider(0.0, 0.75, value=0.5, step=0.05,
                                           label="Overlap (solapamiento entre ventanas)")
                    sw_btn     = gr.Button("🔍 Detectar alimentos", variant="primary")

                with gr.Column(scale=1):
                    sw_log  = gr.Textbox(label="Alimentos detectados", lines=10,
                                         interactive=False, elem_classes="log-box")
                    sw_img_result = gr.Image(label="Resultado con detecciones", height=350)

            sw_btn.click(
                fn=detectar_en_heladera,
                inputs=[sw_img, sw_filas, sw_cols, sw_umbral, sw_overlap],
                outputs=[sw_log, sw_img_result],
            )

        # ══════════════════════════════════════════════════
        #  TAB 6 — AYUDA / DOCS
        # ══════════════════════════════════════════════════
        with gr.TabItem("📖  Ayuda"):
            gr.Markdown("""
## Flujo de trabajo sugerido

1. **Scraper** — elegí las clases que querés detectar y descargá imágenes.
2. **Dataset** — revisá la distribución; asegurate de tener al menos 20–30 imgs por clase.
3. **Entrenar** — empezá con MobileNetV3-Small, 5 épocas, sin freeze. Si la accuracy es baja, subí épocas o activá augmentation.
4. **Probar** — subí fotos reales de tu heladera y verificá las predicciones.

---

## Estructura de carpetas generada

```
proyecto/
├── dataset_heladera/
│   ├── lechuga/          ← imágenes de lechuga
│   ├── tomate/
│   └── heladera_abierta…/
├── modelos_guardados/
│   ├── mobilenet_v3_small.pth
│   └── ultimo_entrenamiento.json   ← metadatos (clases, ruta)
└── outputs/
    ├── dataset_dist.png
    ├── training_plot.png
    └── inferencia_plot.png
```

---

## Instalación rápida (VPS / local)

```bash
pip install gradio torch torchvision duckduckgo-search requests Pillow matplotlib
python mobilenet_kitchen.py
```

Para acceso público desde VPS:
```bash
python mobilenet_kitchen.py --share
```

---

## Modelos disponibles

| Modelo | Tamaño | Velocidad | Precisión |
|---|---|---|---|
| MobileNetV3-Small | ~10 MB | ⚡⚡⚡ | ★★☆ |
| MobileNetV3-Large | ~22 MB | ⚡⚡☆ | ★★★ |
| MobileNetV2       | ~14 MB | ⚡⚡☆ | ★★☆ |
| EfficientNet-B0   | ~21 MB | ⚡⚡☆ | ★★★ |
""")

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true", help="Crear túnel público (VPS)")
    parser.add_argument("--port",   type=int, default=7860, help="Puerto local")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("  🍽️  MobileNet Kitchen Framework  v2.0")
    print(f"  Dataset: {os.path.abspath(DATASET_ROOT)}")
    print(f"  Modelos: {os.path.abspath(MODELS_DIR)}")
    print(f"  Dispositivo: {'CUDA ✅' if torch.cuda.is_available() else 'CPU (CUDA no disponible)'}")
    print("=" * 60)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
