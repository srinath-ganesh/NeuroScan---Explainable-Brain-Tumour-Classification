'''
import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm # <-- Just added this import

# Import directly from your existing project files
from pipeline import load_sam_model, get_imagenet_transform
from config import CLASS_NAMES
from utils import get_device

def normalize_label(folder_name: str) -> str:
    """Maps all the weird Kaggle folder variations to your strict class names."""
    name = folder_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if "glioma" in name: return "glioma"
    if "meningioma" in name: return "meningioma"
    if "pituitary" in name: return "pituitary"
    if "notumor" in name or "normal" in name: return "notumor"
    return None

def evaluate_multiple_datasets(master_directory):
    device = get_device()
    print(f"Loading SAM model into {device}...\n")
    model = load_sam_model(device)
    transform = get_imagenet_transform()
    
    master_path = Path(master_directory)
    if not master_path.exists():
        print(f"Error: {master_directory} not found.")
        return

    # Loop through the top-level dataset folders
    for dataset_dir in sorted(master_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "Downloaded Zips":
            continue
            
        print(f"{'='*60}")
        print(f"🧪 EVALUATING: {dataset_dir.name}")
        print(f"{'='*60}")

        correct_predictions = 0
        total_images = 0
        
        # Track stats dynamically based on what's actually found
        class_stats = {c.lower().replace("_", ""): {"correct": 0, "total": 0} for c in CLASS_NAMES}

        # 1. Convert the generator to a list so tqdm knows the total number of files for the ETA
        all_files = list(dataset_dir.rglob("*.*"))

        # 2. Wrap your exact original loop with tqdm
        for img_path in tqdm(all_files, desc=f"Evaluating {dataset_dir.name}", unit="img"):
            # 1. Skip non-images
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            # 2. CRITICAL: Skip 'Training' folders to prevent data leakage!
            if "training" in [part.lower() for part in img_path.parts]:
                continue
            
            # 3. Identify the ground truth based on the parent folder's name
            folder_name = img_path.parent.name
            actual_class = normalize_label(folder_name)
            
            # Skip if it's in a random folder that doesn't match a class
            if not actual_class:
                continue 

            total_images += 1
            class_stats[actual_class]["total"] += 1
            
            try:
                # 4. Inference
                pil_img = Image.open(img_path).convert("RGB")
                x = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(x)
                    pred_idx = int(logits.argmax(dim=1)[0])
                
                predicted_class = CLASS_NAMES[pred_idx].lower().replace("_", "")
                
                # 5. Score it
                if predicted_class == actual_class:
                    correct_predictions += 1
                    class_stats[actual_class]["correct"] += 1
                    
            except Exception as e:
                pass # Silently skip corrupted images
        
        # --- Print Results for this specific Dataset ---
        if total_images == 0:
            print("No valid testing images found in this folder.\n")
            continue

        overall_acc = (correct_predictions / total_images) * 100
        # Added a \n here so the progress bar doesn't overwrite the final output
        print(f"\n🏆 Overall Accuracy: {overall_acc:.2f}% ({correct_predictions}/{total_images})\n")
        
        for class_name, stats in class_stats.items():
            if stats["total"] > 0:
                acc = (stats["correct"] / stats["total"]) * 100
                print(f" - {class_name.capitalize():<12}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        print("\n")

if __name__ == "__main__":
    evaluate_multiple_datasets("Kaggle_Mega_Test")
'''

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import directly from your existing project files
from pipeline import load_sam_model, get_imagenet_transform
from config import CLASS_NAMES
from utils import get_device

def normalize_label(folder_name: str) -> str:
    """Maps all the weird Kaggle folder variations to your strict class names."""
    name = folder_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if "glioma" in name: return "glioma"
    if "meningioma" in name: return "meningioma"
    if "pituitary" in name: return "pituitary"
    if "notumor" in name or "normal" in name: return "notumor"
    return None

class MRIDataset(Dataset):
    """Custom PyTorch Dataset to handle loading images asynchronously."""
    def __init__(self, file_mapping, transform):
        self.file_mapping = file_mapping
        self.transform = transform

    def __len__(self):
        return len(self.file_mapping)

    def __getitem__(self, idx):
        img_path, actual_class = self.file_mapping[idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
            x = self.transform(pil_img)
            return x, actual_class, True # True means it loaded successfully
        except Exception:
            # Return a dummy tensor if the image is corrupted
            return torch.zeros((3, 224, 224)), actual_class, False

def evaluate_multiple_datasets(master_directory):
    device = get_device()
    print(f"Loading SAM model into {device}...\n")
    model = load_sam_model(device)
    transform = get_imagenet_transform()
    
    master_path = Path(master_directory)
    if not master_path.exists():
        print(f"Error: {master_directory} not found.")
        return

    # Loop through the top-level dataset folders
    for dataset_dir in sorted(master_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "Downloaded Zips":
            continue
            
        print(f"{'='*60}")
        print(f"🧪 EVALUATING: {dataset_dir.name}")
        print(f"{'='*60}")

        # 1. Pre-filter valid files to feed to the Dataset
        valid_files = []
        for img_path in dataset_dir.rglob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            if "training" in [part.lower() for part in img_path.parts]:
                continue
            
            actual_class = normalize_label(img_path.parent.name)
            if actual_class:
                valid_files.append((img_path, actual_class))
        
        if not valid_files:
            print("No valid testing images found in this folder.\n")
            continue

        # 2. Setup DataLoader for hardware-accelerated batching
        dataset = MRIDataset(valid_files, transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=64,   # Send 64 images to the MPS at once
            num_workers=4,   # Use 4 background processes to load images
            shuffle=False
        )

        correct_predictions = 0
        total_images = 0
        class_stats = {c.lower().replace("_", ""): {"correct": 0, "total": 0} for c in CLASS_NAMES}

        # 3. Batched Inference Loop
        for batch_x, batch_actual_classes, batch_valid in tqdm(dataloader, desc=f"Evaluating {dataset_dir.name}"):
            batch_x = batch_x.to(device)
            
            with torch.no_grad():
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
            
            # Unpack the batch results
            for i in range(len(preds)):
                if not batch_valid[i]:
                    continue # Skip corrupted images
                    
                pred_idx = int(preds[i])
                predicted_class = CLASS_NAMES[pred_idx].lower().replace("_", "")
                actual_class = batch_actual_classes[i]
                
                total_images += 1
                class_stats[actual_class]["total"] += 1
                
                if predicted_class == actual_class:
                    correct_predictions += 1
                    class_stats[actual_class]["correct"] += 1
        
        # --- Print Results ---
        overall_acc = (correct_predictions / total_images) * 100 if total_images > 0 else 0
        print(f"\n🏆 Overall Accuracy: {overall_acc:.2f}% ({correct_predictions}/{total_images})\n")
        
        for class_name, stats in class_stats.items():
            if stats["total"] > 0:
                acc = (stats["correct"] / stats["total"]) * 100
                print(f" - {class_name.capitalize():<12}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        print("\n")

if __name__ == "__main__":
    evaluate_multiple_datasets("Kaggle_Mega_Test")