
import os

def parse_metadata_file(filepath):
    metadata = {
        "chart_type": None,
        "title": None,
        "x_axis": None,
        "y_axis": None,
        "source": None
    }

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip() if value.strip().lower() != 'null' else None

                key_map = {
                    "chart type": "chart_type",
                    "title": "title",
                    "x axis": "x_axis",
                    "y axis": "y_axis",
                    "source": "source"
                }

                if key in key_map:
                    metadata[key_map[key]] = value
    return metadata


def load_all_metadata(folder_path):
    all_metadata = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            image_id = os.path.splitext(filename)[0]
            metadata = parse_metadata_file(os.path.join(folder_path, filename))
            all_metadata[image_id] = metadata
    return all_metadata

if __name__ == "__main__":
    folder_path = "../metadata_folder" 
    all_metadata = load_all_metadata(folder_path)
    
    for image_id, metadata in all_metadata.items():
        print(f"\nImage ID: {image_id}")
        for k, v in metadata.items():
            print(f"{k}: {v}")
