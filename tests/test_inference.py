import numpy as np
from PIL import Image

from src.inference import overlay_attention_heatmap


def test_overlay_attention_heatmap_output_shape():
    base_img = Image.new("RGB", (100, 100), color="blue")
    att_map = np.random.rand(7, 7).astype(np.float32)

    result = overlay_attention_heatmap(base_img, att_map, alpha=0.5, colormap_name="jet")

    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)
    assert result.mode == "RGB"

