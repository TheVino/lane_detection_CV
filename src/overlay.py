# src/overlay.py
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def add_custom_text(image, text, position, font_path, font_size, color=(255, 255, 255), shadow_color=(0, 0, 0)):
    # Convert image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    
    # Add shadow
    shadow_offset = 2
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), text, font=font, fill=shadow_color)

    # Add main text
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_icon(frame, icon, position):
    """Overlay an icon on the frame at the specified position."""
    h, w = icon.shape[:2]
    x, y = position

    # Ensure the overlay doesn't go out of bounds
    if x + w > frame.shape[1] or y + h > frame.shape[0]:
        print("Icon position is out of bounds.")
        return frame

    # Resize the icon to fit within the frame dimensions
    icon = resize_icon(icon, max_width=200)  # Resize icon to fit

    # Extract the alpha channel for blending
    if icon.shape[2] == 4:  # RGBA image
        alpha = icon[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        for c in range(3):  # Iterate over RGB channels
            frame[y:y+icon.shape[0], x:x+icon.shape[1], c] = (
                alpha * icon[:, :, c] + (1 - alpha) * frame[y:y+icon.shape[0], x:x+icon.shape[1], c]
            )
    else:  # No alpha channel, directly overlay
        frame[y:y+icon.shape[0], x:x+icon.shape[1]] = icon

    return frame

def resize_icon(icon, max_width):
    """Resize the icon to fit within the frame."""
    aspect_ratio = icon.shape[1] / icon.shape[0]
    new_width = min(icon.shape[1], max_width)
    new_height = int(new_width / aspect_ratio)
    resized_icon = cv2.resize(icon, (new_width, new_height))
    return resized_icon
