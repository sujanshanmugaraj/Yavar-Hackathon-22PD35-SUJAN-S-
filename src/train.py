
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

from dataset_loader import ImageCaptionDataset
from preprocessor import metadata_to_prompt
from model_loader import load_model_and_processor

torch.set_num_threads(1)

def collate_fn(batch):
    images, metadatas, captions = zip(*batch)
    return images, metadatas, captions

def train_loop(
    model,
    processor,
    dataloader,
    optimizer,
    scheduler,
    device,
    epochs=3,
    save_dir="checkpoints"
):
    model.train()
    ignore_index = processor.tokenizer.pad_token_id  
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        epoch_correct_tokens = 0
        epoch_total_tokens = 0

        for images, metadatas, captions in tqdm(dataloader):
            combined_texts = [
                metadata_to_prompt(md) + " " + caption for md, caption in zip(metadatas, captions)
            ]

            inputs = processor(
                images=images,
                text=combined_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            logits = outputs.logits  
            preds = torch.argmax(logits, dim=-1)  
            labels = inputs["input_ids"]

            mask = labels != ignore_index
            correct = (preds == labels) & mask
            batch_correct_tokens = correct.sum().item()
            batch_total_tokens = mask.sum().item()

            epoch_correct_tokens += batch_correct_tokens
            epoch_total_tokens += batch_total_tokens

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct_tokens / epoch_total_tokens if epoch_total_tokens > 0 else 0

        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} token accuracy: {accuracy:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, default="img_folder")
    parser.add_argument("--metadata_folder", type=str, default="metadata_folder")
    parser.add_argument("--captions_file", type=str, default="captions.json")
    parser.add_argument("--batch_size", type=int, default=2)  
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset_size", type=int, default=30, help="Use a subset of the dataset for faster testing")
    args = parser.parse_args()

    image_transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.Lambda(lambda x: x.convert("RGB"))
    ])

    dataset = ImageCaptionDataset(
        img_folder=args.img_folder,
        metadata_folder=args.metadata_folder,
        captions_file=args.captions_file,
        transform=image_transform
    )

    if args.subset_size > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.subset_size, len(dataset)))))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    processor, model, device = load_model_and_processor()
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(dataloader) * args.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    train_loop(model, processor, dataloader, optimizer, scheduler, args.device, epochs=args.epochs)
