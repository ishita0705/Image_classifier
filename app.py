"""
HuggingFace Image Classifier — Gradio App
Model: google/vit-base-patch16-224 (Vision Transformer)
Run: python app.py
"""

# ── Step 2: Imports ────────────────────────────────────────────────────────────
from transformers import pipeline
from PIL import Image
import gradio as gr
import requests
import torch
import time
from io import BytesIO

# ── Step 3: Load the Model ────────────────────────────────────────────────────
print("Loading model... (first run downloads ~350MB, cached after)")

device = 0 if torch.cuda.is_available() else -1  # use GPU if available, else CPU

classifier = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
    device=device,
    top_k=5  # return top 5 predictions
)

print(f"✅ Model loaded on {'GPU' if device == 0 else 'CPU'}")


# ── Step 4: Core Classification Function ──────────────────────────────────────
def classify_image(image: Image.Image) -> tuple[dict, str]:
    """
    Takes a PIL Image, returns:
      - dict of {label: confidence} for Gradio Label component
      - markdown string with formatted results
    """
    if image is None:
        return {}, "⚠️ Please upload an image or enter a URL."

    start = time.time()
    results = classifier(image)  # list of [{label, score}, ...]
    elapsed = time.time() - start

    # Format for Gradio Label component (expects {label: confidence_0_to_1})
    label_dict = {r["label"].replace("_", " ").title(): round(r["score"], 4) for r in results}

    # Build a readable markdown summary
    top = results[0]
    top_label = top["label"].replace("_", " ").title()
    top_conf = top["score"] * 100

    lines = [
        f"### 🏆 Top prediction: **{top_label}** ({top_conf:.1f}%)",
        f"*Inference time: {elapsed*1000:.0f}ms*",
        "",
        "| Rank | Label | Confidence |",
        "|------|-------|-----------|",
    ]
    for i, r in enumerate(results, 1):
        label = r["label"].replace("_", " ").title()
        conf = r["score"] * 100
        bar = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
        lines.append(f"| {i} | {label} | `{bar}` {conf:.1f}% |")

    return label_dict, "\n".join(lines)


# ── Step 5: Load Image from URL helper ────────────────────────────────────────
def classify_from_url(url: str) -> tuple[Image.Image, dict, str]:
    """Fetch an image from URL and classify it."""
    if not url.strip():
        return None, {}, "⚠️ Please enter an image URL."
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        labels, summary = classify_image(img)
        return img, labels, summary
    except requests.exceptions.Timeout:
        return None, {}, "❌ Request timed out. Try a different URL."
    except Exception as e:
        return None, {}, f"❌ Could not load image: {str(e)}"


# ── Step 6: Sample Images ──────────────────────────────────────────────────────
SAMPLE_URLS = [
    ("🐕 Dog",       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"),
    ("🐈 Cat",       "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"),
    ("🌹 Flower",    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/1200px-Sunflower_from_Silesia2.jpg"),
    ("🚗 Car",       "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2012_Honda_Accord_LX_sedan_%28Blue%29%2C_front_7.23.12.jpg/1200px-2012_Honda_Accord_LX_sedan_%28Blue%29%2C_front_7.23.12.jpg"),
    ("✈️ Airplane",  "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/A320-231_Air_France_F-GFKJ_400%27_MSN_0359_%288431174942%29.jpg/1200px-A320-231_Air_France_F-GFKJ_400%27_MSN_0359_%288431174942%29.jpg"),
]


# ── Step 7: Build the Gradio UI ───────────────────────────────────────────────
with gr.Blocks(title="🤗 Image Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🤗 HuggingFace Image Classifier
    **Model:** `google/vit-base-patch16-224` (Vision Transformer — trained on ImageNet-21k)
    Upload an image or paste a URL to classify it into 1000 ImageNet categories.
    """)

    # ── Tab 1: Upload ──────────────────────────────────────────────────────────
    with gr.Tab("📁 Upload Image"):
        with gr.Row():
            with gr.Column(scale=1):
                upload_input = gr.Image(type="pil", label="Upload an image")
                upload_btn = gr.Button("🔍 Classify", variant="primary")

            with gr.Column(scale=1):
                upload_labels = gr.Label(num_top_classes=5, label="Top 5 Predictions")
                upload_summary = gr.Markdown(label="Details")

        upload_btn.click(
            fn=classify_image,
            inputs=[upload_input],
            outputs=[upload_labels, upload_summary]
        )

        # Also classify immediately when image changes
        upload_input.change(
            fn=classify_image,
            inputs=[upload_input],
            outputs=[upload_labels, upload_summary]
        )

    # ── Tab 2: URL ─────────────────────────────────────────────────────────────
    with gr.Tab("🔗 Image URL"):
        with gr.Row():
            url_input = gr.Textbox(
                placeholder="https://example.com/image.jpg",
                label="Paste an image URL"
            )
            url_btn = gr.Button("🔍 Fetch & Classify", variant="primary")

        with gr.Row():
            url_image_out = gr.Image(type="pil", label="Fetched Image", interactive=False)

            with gr.Column():
                url_labels = gr.Label(num_top_classes=5, label="Top 5 Predictions")
                url_summary = gr.Markdown(label="Details")

        url_btn.click(
            fn=classify_from_url,
            inputs=[url_input],
            outputs=[url_image_out, url_labels, url_summary]
        )

        # Quick sample buttons
        gr.Markdown("**Try a sample:**")
        with gr.Row():
            for name, url in SAMPLE_URLS:
                gr.Button(name).click(
                    fn=lambda u=url: classify_from_url(u),
                    outputs=[url_image_out, url_labels, url_summary]
                )

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    **Model info:** ViT (Vision Transformer) processes images as 16×16 patches, applies self-attention, and classifies into 1000 ImageNet categories.
    **Source:** [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on HuggingFace Hub.
    """)


# ── Step 8: Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on local network
        server_port=7860,
        share=False,             # set True to get a public shareable link
        show_error=True
    )
