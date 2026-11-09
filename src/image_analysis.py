"""Small, local image-insight helpers used by the Streamlit experience."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageStat


def analyse_image(image_bytes: bytes) -> dict[str, object]:
    """Return useful image facts without sending an upload anywhere else."""
    with Image.open(BytesIO(image_bytes)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        thumbnail = image.copy()
        thumbnail.thumbnail((160, 160))
        statistics = ImageStat.Stat(thumbnail)
        rgb = np.asarray(thumbnail, dtype=np.float32)
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        palette = thumbnail.quantize(colors=4, method=Image.Quantize.MEDIANCUT).convert("RGB")
        colours = palette.getcolors(maxcolors=thumbnail.width * thumbnail.height) or []
        colours.sort(reverse=True)
        dominant_colours = [
            "#{:02X}{:02X}{:02X}".format(*colour) for _, colour in colours[:4]
        ]

    orientation = "Square"
    if width > height:
        orientation = "Landscape"
    elif height > width:
        orientation = "Portrait"

    return {
        "width": width,
        "height": height,
        "orientation": orientation,
        "brightness": float(np.mean(luminance)),
        "contrast": float(np.std(luminance)),
        "mean_rgb": tuple(round(channel) for channel in statistics.mean[:3]),
        "dominant_colours": dominant_colours,
    }


def light_description(brightness: float) -> str:
    if brightness < 70:
        return "Low light"
    if brightness > 185:
        return "Bright"
    return "Balanced"


def quality_assessment(width: int, height: int, brightness: float, contrast: float) -> tuple[str, str]:
    """Give a simple, explainable readiness check for caption generation."""
    pixels = width * height
    if pixels < 80_000:
        return "Limited", "Use a larger, sharper image for more reliable recognition."
    if brightness < 45:
        return "Low light", "Try brighter lighting or a clearer view of the subject."
    if contrast < 12:
        return "Low contrast", "Increase contrast so objects stand out from the background."
    return "Ready", "The image has enough visual detail for captioning."
