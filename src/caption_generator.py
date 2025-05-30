
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
        predicted_token_id = sequences[0, i+1]
        prob = F.softmax(logits, dim=-1)[0, predicted_token_id].item()
        probs.append(prob)

    confidence_score = sum(probs) / len(probs) if probs else 0.0

    return caption, confidence_score
   

