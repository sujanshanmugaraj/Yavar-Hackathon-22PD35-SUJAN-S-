
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

def overlay_caption(
    image_path,
    concise_caption,
    detailed_caption,
    concise_conf,
    detailed_conf,
    output_path,
    font_path=None,
    font_size=24,
    font_color=(255, 255, 255),  
    bg_color=(0, 0, 0, 128),  
    position=("bottom", "center"),
    max_width=60,
    padding=10,
):
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    if font_path and os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    caption = (
        f"Concise: {concise_caption} (Conf: {concise_conf:.2f})\n"
        f"Detailed: {detailed_caption} (Conf: {detailed_conf:.2f})"
    )

    wrapped_text = textwrap.fill(caption, width=max_width)
    lines = wrapped_text.split('\n')

    txt_layer = Image.new('RGBA', image.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)

    bbox = font.getbbox("A")
    line_height = (bbox[3] - bbox[1]) + 4

    text_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
    text_height = line_height * len(lines)

    if position[0] == "bottom":
        y = height - text_height - padding*2
    elif position[0] == "top":
        y = padding
    else:  # center
        y = (height - text_height) // 2

    if position[1] == "left":
        x = padding
    elif position[1] == "right":
        x = width - text_width - padding*2
    else:
        x = (width - text_width) // 2

    rect_x0 = x - padding
    rect_y0 = y - padding
    rect_x1 = x + text_width + padding
    rect_y1 = y + text_height + padding
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=bg_color)

    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, font=font, fill=font_color)

    out = Image.alpha_composite(image, txt_layer).convert("RGB")
    out.save(output_path)
    print(f"🖼️ Caption overlaid and saved to: {output_path}")
