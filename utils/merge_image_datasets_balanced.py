import os
import shutil
import random

def merge_datasets_equally(dataset_dirs, output_dir, total_images=3000):
    os.makedirs(output_dir, exist_ok=True)
    num_sources = len(dataset_dirs)
    images_per_dataset = total_images // num_sources

    all_copied = []

    for i, dataset_dir in enumerate(dataset_dirs):
        images = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)
        selected = images[:images_per_dataset]

        for img in selected:
            src_path = os.path.join(dataset_dir, img)
            new_name = f"ds{i+1}_{img}"  
            dst_path = os.path.join(output_dir, new_name)
            shutil.copy2(src_path, dst_path)
            all_copied.append(new_name)

    print(f"[INFO] Merged {len(all_copied)} images into {output_dir}")

if __name__ == "__main__":
    dataset_paths = [
        r"C:\Users\Sujan.S\Downloads\archive (2)\graphs", 
        r"C:\Users\Sujan.S\Downloads\archive (3)\Images",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\dogs",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\data",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\cats",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\cars",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\bike",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\human",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\horses",
        r"C:\Users\Sujan.S\Downloads\archive (4)\data\flowers"
    ]
    final_output = "img_folder"
    merge_datasets_equally(dataset_paths, final_output, total_images=3000)
