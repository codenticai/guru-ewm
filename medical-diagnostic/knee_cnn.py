"""
knee_cnn.py — compact CNN knee-MRI abnormality classifier (CPU-only).

A small convolutional network (~50k params) that runs fully on CPU for both
training and inference. It replaces the hand-crafted HLLSet fingerprint with
a learned feature extractor.

Training data: a labeled folder of grayscale images:

    data/knee_mri/
      acl_tear/*.png
      meniscal_tear/*.png
      ...

or the built-in synthetic generators (knee_mri.generate_synthetic) via
scripts/train_knee_cnn.py --synthetic.

NOTE: trained on synthetic images this does NOT generalize to real MRI scans;
it exists so that a CPU-only training path is ready for a real labeled dataset.
This is NOT a medical device.
"""

import io

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep in sync with knee_mri.py.
CLASSES = [
    "acl_tear",
    "meniscal_tear",
    "osteoarthritis",
    "patellar_dislocation",
    "bone_marrow_lesion",
    "chondral_defect",
]

LABELS = {
    "acl_tear": "ACL Tear",
    "meniscal_tear": "Meniscal Tear",
    "osteoarthritis": "Osteoarthritis",
    "patellar_dislocation": "Patellar Dislocation",
    "bone_marrow_lesion": "Bone Marrow Lesion",
    "chondral_defect": "Chondral Defect",
}

SEVERITY = {
    "acl_tear": "high",
    "meniscal_tear": "moderate",
    "osteoarthritis": "moderate",
    "patellar_dislocation": "high",
    "bone_marrow_lesion": "moderate",
    "chondral_defect": "moderate",
}

IMG_SIZE = 256


class KneeCNN(nn.Module):
    """Compact 4-block CNN → 6 abnormality logits."""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(raw: bytes) -> torch.Tensor:
    """Image bytes → (1, 1, IMG_SIZE, IMG_SIZE) float tensor in [0, 1]."""
    img = Image.open(io.BytesIO(raw)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict(model: KneeCNN, raw: bytes, top_k: int = 6) -> list:
    """Return ranked findings [{abnormality,label,severity,confidence}]."""
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(preprocess(raw)), dim=1)[0]
    order = torch.argsort(probs, descending=True)
    return [
        {
            "abnormality": CLASSES[i],
            "label": LABELS[CLASSES[i]],
            "severity": SEVERITY[CLASSES[i]],
            "confidence": round(float(probs[i]), 4),
        }
        for i in order[: max(1, min(top_k, len(CLASSES)))]
    ]


def save_model(model: KneeCNN, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, path)


def load_model(path: str) -> KneeCNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KneeCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def train_model(model: KneeCNN, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3) -> dict:
    """Train on CPU. Returns {'train_acc', 'val_acc'}."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    def accuracy(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / max(total, 1)

    return {"train_acc": round(accuracy(train_loader), 4), "val_acc": round(accuracy(val_loader), 4)}
