import os

IMG_FOLDER = "img_folder"

extensions_to_fix = [".jpg.jpg",".png.jpg", ".jpg.jpeg", ".jpeg.jpg", ".jpeg.jpeg"]

for filename in os.listdir(IMG_FOLDER):
    old_path = os.path.join(IMG_FOLDER, filename)
    
    if not os.path.isfile(old_path):
        continue

    for ext in extensions_to_fix:
        if filename.lower().endswith(ext):
            base_name = filename[: -len(ext)]
            new_filename = base_name + ".jpg"
            new_path = os.path.join(IMG_FOLDER, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            break  
