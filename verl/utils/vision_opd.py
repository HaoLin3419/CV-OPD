"""Small image helpers shared by Vision-OPD data paths."""

from PIL import Image, ImageDraw


def add_red_frame(image: Image.Image, width: int = 4) -> Image.Image:
    """Return an RGB copy with a configurable red frame on the outer edge."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL image, got {type(image)}")
    if width <= 0:
        return image.convert("RGB")
    image = image.convert("RGB").copy()
    width = min(int(width), max(1, min(image.size) // 2))
    draw = ImageDraw.Draw(image)
    for offset in range(width):
        draw.rectangle(
            (offset, offset, image.width - 1 - offset, image.height - 1 - offset),
            outline=(255, 0, 0),
        )
    return image
