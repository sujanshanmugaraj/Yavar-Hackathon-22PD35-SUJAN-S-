import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model_and_processor(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load BLIP image captioning base model (BLIP-1).
    Returns processor, model, and device.
    """
    model_name = "Salesforce/blip-image-captioning-base"

    try:
        print(f"🔄 Loading BLIP processor and model: {model_name} ...")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        print(f"🚀 Moving model to device: {device}")
        model.to(device)
        model.eval()

        print("✅ Model and processor loaded successfully.\n")
        return processor, model, device

    except Exception as e:
        print("❌ Error loading model:", str(e))
        exit(1)


def generate_caption(processor, model, device, image, prompt=None):
    """
    Generate a caption for the image using optional prompt.
    If no prompt is given, the model generates a general caption.
    """
    try:
        print("📦 Preparing inputs for caption generation...")
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        print("✏️ Generating caption...")
        out = model.generate(**inputs, max_length=50)

        caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print("❌ Error during caption generation:", str(e))
        return ""


if __name__ == "__main__":
    from PIL import Image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    processor, model, device = load_model_and_processor(device=device)

    # Load image
    image_path = "img_folder/sample1.png"
    try:
        print(f"🖼️ Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print("❌ Error loading image:", str(e))
        exit(1)

    prompt = "Describe the trend shown in the chart."

    # Generate caption
    caption = generate_caption(processor, model, device, image, prompt)

    if caption:
        print("\n🎯 Generated Caption:", caption)
    else:
        print("\n⚠️ No caption was generated.")
