
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

image = "img_folder/sample1.png"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

metadata = (
    "Sales and client acquisitions increased steadily from 2018 to 2022, "
    "with spikes in Q4 (October to December)."
)
prompt = f"{metadata} Describe the chart."

inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

generated_ids = model.generate(**inputs, max_new_tokens=100)
caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\n📊 Generated Caption:\n", caption)
