import os
import json

def parse_metadata_file(file_path):
    """
    Parses a metadata .txt file with lines formatted as:
    key: value
    Returns a dictionary with keys and values.
    """
    metadata = {
        "section_header": "",
        "above_text": "",
        "caption": "",
        "footnote": "",
        "below_text": ""
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower()  # normalize key to lowercase
                val = val.strip()
                if key in metadata:
                    metadata[key] = val
                else:
                    print(f"[WARNING] Unexpected key '{key}' in {file_path}")
    return metadata

def load_dataset(img_folder, metadata_folder):
    """
    Loads all images and their corresponding metadata.

    Returns:
        List of tuples: (image_path, metadata_dict)
    """
    dataset = []
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        meta_filename = os.path.splitext(img_file)[0] + ".txt"
        meta_path = os.path.join(metadata_folder, meta_filename)

        if os.path.exists(meta_path):
            metadata = parse_metadata_file(meta_path)
            dataset.append((img_path, metadata))
        else:
            print(f"[WARNING] Metadata file not found for image: {img_file}")

    return dataset

def save_metadata_as_json(metadata_dict, output_path):
    """
    Saves metadata dictionary as a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)

# Test run example
if __name__ == "__main__":
    img_folder = r"C:\Users\Sujan.S\OneDrive\Documents\GitHub\Image-Captioning-from-Contextual-Metadata-Using-Vision-Language-Models-VLMs-\img_folder"
    metadata_folder = r"C:\Users\Sujan.S\OneDrive\Documents\GitHub\Image-Captioning-from-Contextual-Metadata-Using-Vision-Language-Models-VLMs-\metadata_folder"
    dataset = load_dataset(img_folder, metadata_folder)

    print(f"Loaded {len(dataset)} image-metadata pairs.\n")
    for img_path, metadata in dataset[:3]:
        print(f"Image: {img_path}")
        print("Metadata:", metadata)
        print("-" * 40)
