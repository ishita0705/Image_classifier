# 🤗 HuggingFace Image Classifier

A Gradio web app that classifies images into 1000 categories using Google's Vision Transformer (ViT).

## ⚡ Quick Start

```bash
# Step 1 — Clone / open this folder in VS Code

# Step 2 — Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Step 3 — Install dependencies
pip install -r requirements.txt

# Step 4 — Run the app
python app.py
```

Then open **http://localhost:7860** in your browser.

---

## 📁 Project Structure

```
image_classifier/
├── app.py              ← Main app (all code lives here)
├── requirements.txt    ← Dependencies
└── README.md           ← This file
```

## 🧠 How It Works

1. **Model:** `google/vit-base-patch16-224` — a Vision Transformer pretrained on ImageNet-21k (14M images, 21k classes), fine-tuned on ImageNet-1k (1.2M images, 1000 classes)
2. **Input:** Any image — uploaded file or URL
3. **Process:** Image → resize to 224×224 → split into 16×16 patches → transformer encoder → classification head
4. **Output:** Top 5 predictions with confidence scores

## 🚀 What to Try Next

- **Different model:** Replace `google/vit-base-patch16-224` with `microsoft/resnet-50` or `facebook/convnext-base-224`
- **More categories:** Use `openai/clip-vit-base-patch32` with custom label lists (zero-shot!)
- **Object detection:** Switch `task` to `"object-detection"` with `facebook/detr-resnet-50`
- **Deploy:** Set `share=True` in `demo.launch()` for a public shareable URL
