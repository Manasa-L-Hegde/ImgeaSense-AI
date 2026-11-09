"""ImageSense AI: MobileNetV2 image features with an LSTM caption decoder."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_ROOT / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
