# ImageSense AI — Multimodal CNN + RNN Image Captioning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imgeasense-ai.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![BLEU-1 Score](https://img.shields.io/badge/BLEU--1-54.0%25-brightgreen.svg)](#evaluation-results)

🚀 **Live Interactive Web App:** [https://imgeasense-ai.streamlit.app/](https://imgeasense-ai.streamlit.app/)

ImageSense AI is an end-to-end multimodal deep learning mini-project combining **Computer Vision (CNN)** and **Natural Language Processing (RNN)**:
1. **Pretrained MobileNetV2 (CNN Encoder):** Extracts 49 spatial visual regions ($7 \times 7 \times 1280$) from an input image.
2. **Bahdanau Additive Attention:** Dynamically calculates attention weights, allowing the model to look at the relevant image region for each word.
3. **LSTM Decoder (RNN):** Generates grammatically coherent natural-language captions word-by-word.
4. **Interactive Attention Heatmap & Audio:** Visually inspect which image regions the network focused on for every word, and listen to the caption via integrated Text-to-Speech (TTS).

---

## 🌐 Live Web Demo

You can try the live application directly in your browser without installing anything:  
👉 **[ImageSense AI on Streamlit Cloud](https://imgeasense-ai.streamlit.app/)**

## Setup

```powershell
# TensorFlow is supported on Python 3.12 for this project.
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit app

```powershell
streamlit run app.py
```

The app requires trained artifacts in `artifacts/`. Until the Flickr8k training process finishes, it
shows the training command and keeps caption generation disabled. This project intentionally has no
classifier-template fallback: every generated caption comes from the trained CNN–attention–LSTM model.

## Train on Flickr8k

Obtain Flickr8k from an authorized hosting source and verify its current terms. Arrange files as:

```text
data/
├── Flickr8k_Dataset/
│   └── *.jpg
└── Flickr8k_text/
    └── Flickr8k.token.txt
```

Then run the reproducible training pipeline:

```powershell
python -m src.train --data-dir data --epochs 10 --batch-size 64
```

The pipeline splits by image into 80% train, 10% validation, and 10% test, so captions from the same
image cannot leak between splits. It saves `artifacts/caption_decoder.keras`,
`artifacts/tokenizer.json`, and `artifacts/training_history.npz`. Training is CPU-intensive; use a
GPU-enabled TensorFlow environment when available.

The app activates automatically when the trained CNN–attention–LSTM model and tokenizer are present.

## Evaluate the trained model

After training, evaluate on the reproducible held-out image split:

```powershell
# Fast smoke evaluation
python -m src.evaluate --data-dir data --limit 25

# Full held-out test split
python -m src.evaluate --data-dir data
```

This writes BLEU-1, BLEU-4, and sample predictions to `artifacts/evaluation.json`. See the
[mini-project guide](docs/PROJECT_GUIDE.md) for the architecture, demonstration flow, report table,
and viva-ready notes.

## Test

```powershell
python -m pytest -q
```

## Evaluation Results

| Metric | Measured Score | Description |
| :--- | :--- | :--- |
| **BLEU-1** | **54.0%** | Unigram word accuracy against 5 human reference captions |
| **BLEU-4** | **11.6%** | 4-gram sentence coherence and phrase structure |
| **Training Loss** | **2.428** | Categorical Cross-Entropy on teacher-forced sequences |
| **Validation Loss**| **4.178** | Cross-Entropy error on held-out validation images |
| **Vocabulary** | **1,078 words** | Domain entities, actions, and descriptors |

---

## 📚 Project Documentation & Viva Resources
* 📑 [Comprehensive Project Report (Formal Writeup)](docs/PROJECT_REPORT.md)
* 🧠 [Top 10 Viva Questions & Faculty Cheat Sheet](docs/VIVA_PRESENTATION_GUIDE.md)
* 📊 [Presentation Slides Outline (12 Slides for PPT)](docs/PRESENTATION_SLIDES_OUTLINE.md)
