
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch.nn.functional as F

def load_blip_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_name = "Salesforce/blip-image-captioning-base"
    print(f"🔄 Loading BLIP processor and model: {model_name} ...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print(f"✅ Model and processor loaded on {device}.\n")
    return processor, model, device

def generate_caption_with_confidence(processor, model, image, prompt=None, device='cpu'):
    """
    Generate caption and calculate confidence score from logits.
    Confidence is mean probability of predicted tokens.
    """
    inputs = processor(image, prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=50,
        output_scores=True,
        return_dict_in_generate=True
    )

    sequences = outputs.sequences  
    scores = outputs.scores        

    caption = processor.tokenizer.decode(sequences[0], skip_special_tokens=True)

   
    probs = []
    for i, logits in enumerate(scores):
        logits = logits.cpu()
        predicted_token_id = sequences[0, i+1].item()
        prob = F.softmax(logits, dim=-1)[0, predicted_token_id].item()
        probs.append(prob)

    confidence_score = sum(probs) / len(probs) if probs else 0.0

    print(f"\n📝 Generated Caption: {caption}")
    print(f"🔍 Token Probabilities: {probs}")
    print(f"⭐ Average Confidence: {confidence_score:.4f}\n")

    return caption, confidence_score

if __name__ == "__main__":
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor, model, device = load_blip_model(device)

    image_path = "img_folder/sample1.png"

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"🖼️ Loaded image: {image_path}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)

    prompt = "Describe the trend shown in the chart."

    print("✏️ Generating caption with confidence...")
    caption, confidence = generate_caption_with_confidence(processor, model, image, prompt, device)
