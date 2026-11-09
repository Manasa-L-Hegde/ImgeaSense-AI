# ImageSense AI

ImageSense AI is a CNN + RNN image-captioning prototype based on the project proposal. A pretrained
**MobileNetV2** extracts 49 spatial visual features from an image, and an attention-based **LSTM
decoder** generates a caption word by word. This makes the CNN–RNN connection explicit: the decoder
learns which CNN image regions matter for each caption prefix.

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

## Evaluation notes

Record held-out images, generated captions, and a short human assessment of objects/actions. BLEU can
be calculated against the five Flickr8k reference captions, but it should be reported alongside
qualitative observations because it does not fully measure semantic correctness. Expected limitations
include repetitive captions, incomplete sentences, and incorrect object or action names.
