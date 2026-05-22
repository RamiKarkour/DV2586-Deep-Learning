import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


SEED = 42
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

CUSTOM_EPOCHS = 40
PRETRAINED_EPOCHS = 12
CUSTOM_BATCH_SIZE = 128
PRETRAINED_BATCH_SIZE = 64
CUSTOM_LR = 1e-3
PRETRAINED_LR = 1e-4
WEIGHT_DECAY = 5e-4
CUSTOM_PATIENCE = 8
PRETRAINED_PATIENCE = 6

CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD = [0.2675, 0.2565, 0.2761]


MODEL_NAMES = [
    "resnet50",
    "vgg19_bn",
    "densenet121",
    "efficientnet_b0"
]


DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "resnet50": "ResNet-50",
    "vgg19_bn": "VGG-19",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0"
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "filters"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(F.relu(self.bn6(self.conv6(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def build_model(name):
    if name == "custom_cnn":
        return CustomCNN(NUM_CLASSES)

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    if name == "vgg19_bn":
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        return model

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        return model

    raise ValueError(f"Unknown model: {name}")


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, eval_transform


def build_loaders(batch_size):
    train_transform, eval_transform = get_transforms()

    train_source = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=train_transform)
    val_source = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_set = Subset(train_source, train_indices)
    val_set = Subset(val_source, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def save_filter_plot(weights, path, title, max_filters=8):
    weights = weights.detach().cpu().clone()
    count = min(max_filters, weights.shape[0])

    fig, axes = plt.subplots(1, count, figsize=(count * 1.4, 1.8))
    if count == 1:
        axes = [axes]

    for idx in range(count):
        image = weights[idx]
        image = image - image.min()
        image = image / (image.max() + 1e-8)
        image = image.permute(1, 2, 0).numpy()
        axes[idx].imshow(image)
        axes[idx].axis("off")
        axes[idx].set_title(f"F{idx + 1}", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_history(history, name):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{DISPLAY_NAMES[name]} Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{name}_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Training accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{DISPLAY_NAMES[name]} Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{name}_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(name, model, train_loader, val_loader, epochs, learning_rate, patience):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{DISPLAY_NAMES[name]} | Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(best_state, os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{DISPLAY_NAMES[name]} stopped after epoch {epoch}")
                break

    model.load_state_dict(best_state)
    plot_history(history, name)

    with open(os.path.join(OUTPUT_DIR, "metrics", f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def save_all_histories(histories):
    torch.save(histories, os.path.join(OUTPUT_DIR, "metrics", "training_histories.pt"))

    compact = {}
    for key, value in histories.items():
        compact[key] = {k: [float(x) for x in v] for k, v in value.items()}

    with open(os.path.join(OUTPUT_DIR, "metrics", "training_histories.json"), "w") as f:
        json.dump(compact, f, indent=2)


def main():
    set_seed(SEED)
    make_dirs()

    print(f"Device: {DEVICE}")
    print(f"Mixed precision: {USE_AMP}")

    histories = {}

    train_loader, val_loader, _ = build_loaders(CUSTOM_BATCH_SIZE)
    custom_model = build_model("custom_cnn")

    save_filter_plot(
        custom_model.conv1.weight,
        os.path.join(OUTPUT_DIR, "filters", "custom_cnn_filters_before.png"),
        "Custom CNN first-layer filters before training"
    )

    custom_model, history = train_model(
        "custom_cnn",
        custom_model,
        train_loader,
        val_loader,
        CUSTOM_EPOCHS,
        CUSTOM_LR,
        CUSTOM_PATIENCE
    )
    histories["custom_cnn"] = history

    save_filter_plot(
        custom_model.conv1.weight,
        os.path.join(OUTPUT_DIR, "filters", "custom_cnn_filters_after.png"),
        "Custom CNN first-layer filters after training"
    )

    train_loader, val_loader, _ = build_loaders(PRETRAINED_BATCH_SIZE)

    for name in MODEL_NAMES:
        print(f"\nTraining {DISPLAY_NAMES[name]}")
        model = build_model(name)
        model, history = train_model(
            name,
            model,
            train_loader,
            val_loader,
            PRETRAINED_EPOCHS,
            PRETRAINED_LR,
            PRETRAINED_PATIENCE
        )
        histories[name] = history

    save_all_histories(histories)
    print("Training finished. Run q2_evaluation.py to create final metrics and comparison plots.")


if __name__ == "__main__":
    main()
