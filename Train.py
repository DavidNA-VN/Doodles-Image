import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import pickle

import torch.nn as nn
class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


# ==================== CONFIG ====================
DATA_DIR = "/kaggle/input/datasets/namanhtrnchust/quickdrawdataset/QuickDrawDataset"  # sửa theo path dataset Kaggle
OUTPUT_DIR = "/kaggle/working"

NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
SEED = 42

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_checkpoint.pt")
LAST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "last_checkpoint.pt")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.pkl")

# ==================== SEED ====================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==================== TRANSFORMS ====================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ==================== SPLIT FUNCTION ====================
def get_train_val_indices(dataset, val_split=0.2):
    labels = [label for _, label in dataset.samples]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=SEED
    )

    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx

# ==================== EVALUATE ====================
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc

# ==================== TRAIN ====================
def train():
    # Dataset gốc chỉ dùng để lấy samples + labels
    base_dataset = datasets.ImageFolder(DATA_DIR)

    print("Classes:", base_dataset.classes)
    print("Number of classes:", len(base_dataset.classes))

    counts = {cls: 0 for cls in base_dataset.classes}
    for _, label in base_dataset.samples:
        counts[base_dataset.classes[label]] += 1
    print("Class distribution:", counts)

    train_idx, val_idx = get_train_val_indices(base_dataset, VAL_SPLIT)

    # Tạo 2 dataset riêng để transform không đè nhau
    train_full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    train_dataset = Subset(train_full_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = MobileNetV1(ch_in=3, n_classes=NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_train = 0

        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_train += labels.size(0)

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total_train
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print("-" * 50)
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.2f}%")
        print("-" * 50)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
            "classes": base_dataset.classes,
            "history": history
        }, LAST_CHECKPOINT_PATH)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "classes": base_dataset.classes,
                "history": history
            }, CHECKPOINT_PATH)

            print(f"Saved best checkpoint: {CHECKPOINT_PATH}")
            print(f"Best Val Acc: {best_val_acc:.2f}%")

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)

    print("Training completed.")
    print("Best checkpoint:", CHECKPOINT_PATH)
    print("Last checkpoint:", LAST_CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    train()