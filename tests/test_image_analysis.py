from io import BytesIO

from PIL import Image

from src.image_analysis import analyse_image, light_description, quality_assessment


def test_analyse_image_returns_local_visual_facts():
    image = Image.new("RGB", (40, 20), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    result = analyse_image(buffer.getvalue())

    assert result["width"] == 40
    assert result["height"] == 20
    assert result["orientation"] == "Landscape"
    assert result["dominant_colours"]
    assert light_description(float(result["brightness"])) == "Bright"


def test_quality_assessment_explains_common_image_limits():
    assert quality_assessment(100, 100, 120, 40)[0] == "Limited"
    assert quality_assessment(1000, 800, 20, 40)[0] == "Low light"
    assert quality_assessment(1000, 800, 120, 5)[0] == "Low contrast"
    assert quality_assessment(1000, 800, 120, 40)[0] == "Ready"
