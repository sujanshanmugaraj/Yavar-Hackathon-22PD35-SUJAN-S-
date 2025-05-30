import os
from PIL import Image
import torchvision.transforms as transforms

def load_image(img_folder, image_id):
    # Supports .jpg, .png
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(img_folder, image_id + ext)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"No image found for {image_id} in {img_folder}")

def preprocess_image(image, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image)
