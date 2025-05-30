
import os
from PIL import Image
from torch.utils.data import Dataset
import json

class ImageCaptionDataset(Dataset):
    def __init__(self, img_folder, metadata_folder, captions_file=None, transform=None):
        self.img_folder = img_folder
        self.metadata_folder = metadata_folder
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(img_folder) if f.lower().endswith('.png')
        ])

        self.metadata = {}
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            meta_path = os.path.join(metadata_folder, base_name + '.txt')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    meta_dict = {}
                    for line in lines:
                        if ':' in line:
                            key, val = line.strip().split(':', 1)
                            meta_dict[key.strip()] = val.strip()
                    self.metadata[img_file] = meta_dict
            else:
                self.metadata[img_file] = {}

        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r', encoding='utf-8') as f:
                self.captions = json.load(f)
            print(f"[INFO] Loaded {len(self.captions)} captions from {captions_file}")
        else:
            self.captions = {}
            if captions_file:
                print(f"[WARNING] Captions file {captions_file} not found, proceeding without captions")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_folder, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        metadata = self.metadata.get(img_name, {})
        caption = self.captions.get(img_name, "") 

        return image, metadata, caption


if __name__ == "__main__":
    from torchvision import transforms

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageCaptionDataset(
        img_folder='img_folder',
        metadata_folder='metadata_folder',
        captions_file='captions.json',  
        transform=image_transform
    )

    print(f"Total samples: {len(dataset)}")

    sample_img, sample_metadata, sample_caption = dataset[0]
    print("Sample image tensor shape:", sample_img.shape)
    print("Sample metadata:", sample_metadata)
    print("Sample caption:", sample_caption)
