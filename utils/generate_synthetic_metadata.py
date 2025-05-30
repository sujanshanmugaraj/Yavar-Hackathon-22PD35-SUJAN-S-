
import os
import cv2
import pytesseract
import re
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_clean_text(text):
    return bool(re.match(r'^[a-zA-Z0-9\s:.,()%$€₹-]{3,}$', text))

def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_blocks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[❌] Failed to read image: {image_path}")
        return []

    preprocessed = preprocess_image_for_ocr(image)

    config = r'--oem 3 --psm 4'
    data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)

    lines = []
    print(f"[DEBUG] OCR lines for {Path(image_path).name}:")
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        try:
            conf = int(float(data['conf'][i]))
        except:
            conf = -1
        print(f"  OCR line [{conf}%]: {text}")
        if text and conf > 35 and is_clean_text(text):
            lines.append((data['top'][i], text))

    lines.sort(key=lambda x: x[0])

    print("Final extracted lines:")
    for _, line in lines:
        print(f"  {line}")

    return [t[1] for t in lines]

def infer_metadata(lines):
    metadata = {
        "section_header": None,
        "above_text": None,
        "caption": None,
        "below_text": None,
        "footnote": None,
        "picture_id": "#/pictures/0"
    }

    for line in lines:
        lower_line = line.lower()

        if 'source:' in lower_line:
            if not metadata["footnote"]:
                metadata["footnote"] = line
        elif any(k in lower_line for k in ('figure', 'chart', 'caption')):
            if not metadata["caption"]:
                metadata["caption"] = line
        elif not metadata["section_header"]:
            metadata["section_header"] = line
        elif not metadata["above_text"]:
            metadata["above_text"] = line
        elif not metadata["below_text"]:
            metadata["below_text"] = line

    return metadata

def write_metadata_file(metadata_dict, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for key, value in metadata_dict.items():
            f.write(f"{key}: {value if value is not None else 'None'}\n")

def generate_actual_metadata(img_folder, metadata_folder):
    os.makedirs(metadata_folder, exist_ok=True)
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img in image_files:
        img_path = os.path.join(img_folder, img)
        lines = extract_text_blocks(img_path)
        metadata = infer_metadata(lines)
        
        print(f"\nMetadata for {img}:")
        for key, value in metadata.items():
            print(f"{key}: {value if value is not None else 'None'}")
        print("-" * 50)  
        
        txt_filename = Path(img).stem + ".txt"
        txt_path = os.path.join(metadata_folder, txt_filename)
        write_metadata_file(metadata, txt_path)

    print(f"[✅] Metadata generated for {len(image_files)} images in '{metadata_folder}'.")

if __name__ == "__main__":
    generate_actual_metadata("img_folder", "metadata_folder")
