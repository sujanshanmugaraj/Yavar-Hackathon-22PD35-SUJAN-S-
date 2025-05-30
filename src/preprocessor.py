
from PIL import Image
from torchvision import transforms
import torch

def metadata_to_prompt(metadata: dict) -> str:
    """
    Convert structured metadata into a natural language prompt.
    """
    prompt_parts = []

    if metadata.get("section_header"):
        prompt_parts.append(f"Section: {metadata['section_header']}.")
    if metadata.get("above_text"):
        prompt_parts.append(f"Above: {metadata['above_text']}.")
    if metadata.get("caption"):
        prompt_parts.append(f"Caption: {metadata['caption']}.")
    if metadata.get("below_text"):
        prompt_parts.append(f"Below: {metadata['below_text']}.")
    if metadata.get("footnote"):
        prompt_parts.append(f"Footnote: {metadata['footnote']}.")

    return " ".join(prompt_parts)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image from the given path.
    """
    image = Image.open(image_path).convert("RGB")
    return image_transform(image)

if __name__ == "__main__":
    sample_metadata = {
        "section_header": "Performance Trends",
        "above_text": "Annual summary",
        "caption": "Sales growth trend from 2018 to 2022",
        "footnote": "Q4 spikes due to seasonal demand.",
        "below_text": "Source - Internal Report"
    }
    prompt = metadata_to_prompt(sample_metadata)
    print("Generated Prompt:")
    print(prompt)

    sample_img_path = "img_folder/sample1.png"

    try:
        img_tensor = preprocess_image(sample_img_path)
        print(f"\nPreprocessed image tensor shape: {img_tensor.shape}")
    except FileNotFoundError:
        print(f"\nImage file not found at path: {sample_img_path}. Please check the path.")
