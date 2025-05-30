import os
import json

img_folder = 'img_folder'
output_json = 'captions.json'

image_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.png')]
captions = {}

for img_file in image_files:
    captions[img_file] = f"Caption for {img_file}"

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(captions, f, indent=4)

print(f"Dummy captions JSON created with {len(captions)} entries at {output_json}")

