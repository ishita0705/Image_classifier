# 🖼️ Image Classifier using HuggingFace Transformers + Gradio

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue?style=flat)](https://huggingface.co/spaces/BunnyCode/image_classifier)

Classify any image into 1000 ImageNet categories using Google's Vision Transformer (ViT) — with a live Gradio web UI. Supports image upload and URL input. No model training required.

🚀 **[Try the live demo →](https://huggingface.co/spaces/BunnyCode/image_classifier)**

---

## 📸 Screenshots

**Classify from URL:**
![Uploading image_url.png…]()



**Upload an image:**

<img width="1918" height="1086" alt="image" src="https://github.com/user-attachments/assets/04a09a30-45d3-4efe-b567-efde354ec47b" />


---

## 📌 Overview

This project builds an **image classification web app** using a pretrained Vision Transformer (ViT) model from HuggingFace's model hub. It takes any image as input — either uploaded directly or fetched from a URL — and returns the **top 5 predicted labels** with confidence scores and inference time.

The app is deployed on **HuggingFace Spaces** and runs entirely in the browser — no installation needed to try it.

### What this project demonstrates
- Loading a state-of-the-art pretrained vision model in a few lines of code
- Running image classification inference using HuggingFace `pipeline()`
- Fetching and classifying images from URLs
- Building a multi-tab interactive web UI with Gradio
- Deploying a live ML app on HuggingFace Spaces

---

## 🤗 Where HuggingFace is Used

| Level | What it does |
|-------|-------------|
| `transformers` library | Provides the `pipeline()` API for image classification |
| HuggingFace Hub | Hosts the ViT model weights — downloaded automatically on first run |
| The model itself | `google/vit-base-patch16-224` — Vision Transformer by Google, hosted on HuggingFace |
| HuggingFace Spaces | Free hosting platform where the live demo runs |

---

## 🧠 Model Details

| Property | Detail |
|----------|--------|
| Model | `google/vit-base-patch16-224` |
| Architecture | Vision Transformer (ViT) — processes images as 16×16 patches |
| Pretrained on | ImageNet-21k (14M images, 21k classes) |
| Fine-tuned on | ImageNet-1k (1.2M images, 1000 classes) |
| Model size | ~350MB (downloaded and cached on first run) |
| Output | Top-5 labels from 1000 ImageNet categories |
| HuggingFace Hub | [View model →](https://huggingface.co/google/vit-base-patch16-224) |

---

## 🖥️ How It Works

```
Input Image (upload or URL)
        ↓
Resize to 224×224
        ↓
Split into 16×16 patches
        ↓
Transformer encoder (self-attention across patches)
        ↓
Classification head
        ↓
Top-5 predictions with confidence scores
```

---

## 💻 Sample Output

```
Top prediction: Planetarium (70.9%)
Inference time: 337ms

Rank  Label                                   Confidence
1     Planetarium                             70.9%
2     Cinema, Movie Theater, Movie House      17.0%
3     Theater Curtain, Theatre Curtain         3.0%
4     Stage                                    1.8%
5     Home Theater, Home Theatre               1.2%
```

---

## 📁 Project Structure

```
image-classifier/
├── app.py              ← Main Gradio app (all code lives here)
├── requirements.txt    ← All dependencies
└── README.md           ← You are here
```

---

## 🚀 Run Locally

**Step 1 — Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the app**
```bash
python app.py
```

**Step 4 — Open in browser**
```
http://localhost:7860
```

> First run downloads the ViT model (~350MB) from HuggingFace Hub. Cached after that.

---

## 🗂️ App Features

| Feature | Description |
|---------|-------------|
| Upload tab | Drag and drop any image — classifies instantly on upload |
| URL tab | Paste any image URL — fetches and classifies in one click |
| Sample buttons | One-click test images: Dog, Cat, Flower, Car, Airplane |
| Top 5 predictions | Shows all top-5 labels with confidence bars |
| Inference time | Displays how long the model took in milliseconds |
| Confidence table | Ranked table with visual progress bars per label |

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `transformers` | HuggingFace — model loading and inference |
| `torch` | PyTorch backend for neural network execution |
| `torchvision` | Image preprocessing utilities |
| `gradio` | Web UI framework |
| `Pillow` | Image loading and processing |
| `requests` | Fetching images from URLs |
| `timm` | Vision model support |

---

## 🔭 What to Try Next

- **Different model** — swap to `microsoft/resnet-50` or `facebook/convnext-base-224` for comparison
- **Zero-shot classification** — use `openai/clip-vit-base-patch32` with your own custom label lists
- **Object detection** — change `task` to `"object-detection"` with `facebook/detr-resnet-50`

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
Made with 🤗 HuggingFace Transformers + Gradio • <a href="https://huggingface.co/spaces/BunnyCode/image_classifier">Live Demo</a>
</div>
