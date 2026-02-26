import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --------------------------
# MODEL DEFINITION
# --------------------------
# Using smp.Unet with resnet34 backbone

# --------------------------
# DATASET
# --------------------------
class RoofDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Only include images that have corresponding masks
        self.images = [f for f in os.listdir(image_dir) if os.path.exists(os.path.join(mask_dir, f.replace(".jpg", ".png")))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = mask.astype("float32") / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask

# --------------------------
# METRICS
# --------------------------
def calculate_iou(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()

# --------------------------
# TRAINING LOOP
# --------------------------
def train_model(epochs=50, lr=1e-4, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

    full_dataset = RoofDataset("images", "masks", transform=transform)
    if len(full_dataset) < 2:
        print("Error: Not enough data to train (need at least 2 images with masks).")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    # Load existing model if it exists
    model_path = "roof_segmentation_model.pth"
    if os.path.exists(model_path):
        print(f"Loading existing weights from {model_path} for finetuning...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_iou = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)

        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), model_path)
            print(f"Checkpoint saved: Improved Val IoU to {best_iou:.4f}")

if __name__ == "__main__":
    train_model(epochs=30)
