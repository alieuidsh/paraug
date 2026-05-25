"""02 — classification: Dataset + DataLoader + train loop with batch-GPU aug.

This is the "Where to put paraug" pattern from the README, end-to-end:
- Dataset only loads + resizes (no per-sample aug — paraug is GPU-batch-native).
- DataLoader ships CPU batches to the train loop.
- Train loop moves the batch to GPU and applies paraug to the whole batch.
- Forward / backward / step as usual.

On a 5060 Ti at bs=32 canvas=224×224 this is ~3× faster than putting
paraug inside `Dataset.__getitem__` (per-sample CPU aug).

Run:
    pip install paraug>=0.6.1
    python examples/02_classification.py

(Uses synthetic data so the example is dependency-free.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from paraug import AugPipeline


# ─── synthetic dataset ────────────────────────────────────────────────
class SyntheticImageDataset(Dataset):
    """Stand-in for a real ImageFolder / WebDataset / etc. — just returns
    a (C, H, W) float tensor in [0, 1] and an integer label."""

    def __init__(self, n=512, num_classes=10, hw=(224, 224)):
        self.n = n
        self.num_classes = num_classes
        self.hw = hw
        # Pre-generate so __getitem__ is pure CPU work (mimicking disk-load).
        g = torch.Generator(device="cpu").manual_seed(0)
        self.imgs = torch.rand(n, 3, *hw, generator=g)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Pure load + (notional) resize. NO paraug here — keep aug on GPU.
        return self.imgs[idx], int(self.labels[idx])


# ─── tiny model ───────────────────────────────────────────────────────
class TinyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── data ─────────────────────────────────────────────────────────
    dataset = SyntheticImageDataset(n=512, num_classes=10, hw=(224, 224))
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=0)   # num_workers>0 also fine; aug doesn't run there

    # ─── augmentation (built ONCE, lives on the train side) ──────────
    aug = AugPipeline({
        "geometric": {
            "affine":      {"p": 1.0, "rot_deg": 15.0,
                              "scale_range": (0.9, 1.1)},
            "perspective": {"p": 0.5, "scale_range": (0.05, 0.10)},
        },
        "photometric": {
            "color_jitter":  {"p": 0.5, "brightness": 0.2, "contrast": 0.2},
            "gamma":         {"p": 0.5, "gamma_range": (0.8, 1.2)},
            "gaussian_blur": {"p": 0.3, "sigma_range": (0.5, 1.5)},
        },
    })
    # On a CUDA box, opt into the fast-noise path for ~1.85× speedup at
    # large canvases (no effect on this small example).
    if torch.cuda.is_available():
        import paraug
        paraug.set_fast_noise(True)

    # ─── model + optimiser ───────────────────────────────────────────
    model = TinyClassifier(num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ─── train loop ──────────────────────────────────────────────────
    model.train()
    EPOCHS = 2
    for epoch in range(EPOCHS):
        total, correct, loss_sum = 0, 0, 0.0
        for step, (images, labels) in enumerate(loader):
            # 1. Move batch to GPU.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 2. Augment the GPU batch in one shot. paraug returns a
            #    (img, mask) tuple — `_` discards the (unused) mask slot.
            images, _ = aug(images, seed_base=42, epoch=epoch, step=step)

            # 3. Forward / loss / backward / step — standard.
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
        print(f"epoch {epoch}  loss {loss_sum/total:.4f}  acc {correct/total:.3f}")


if __name__ == "__main__":
    main()
