import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(ROOT_DIR / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))


@dataclass(frozen=True)
class Settings:
    image_size: tuple[int, int] = (224, 224)
    max_length: int = 34
    embedding_dim: int = 256
    lstm_units: int = 512
    feature_dim: int = 1280
    feature_regions: int = 49
    model_dir: Path = ROOT_DIR / "artifacts"
    model_file: str = "caption_decoder.keras"
    tokenizer_file: str = "tokenizer.json"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / self.tokenizer_file


SETTINGS = Settings()
