import time
import torch
from src import config
from src.models import load_checkpoint

MODELS = [
    "mobilenet_v2_fine_tune.pt",
    "resnet18_full.pt",
    "custom_cnn_scratch.pt",
]

x = torch.randn(1, 3, 224, 224)

for fname in MODELS:
    path = config.MODEL_DIR / fname
    model, name, _ = load_checkpoint(path, device="cpu")
    with torch.no_grad():
        for _ in range(10):  # warm-up
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(x)
        elapsed_ms = ((time.perf_counter() - t0) / 50) * 1000
    print(f"{fname:30s} | CPU inference: {elapsed_ms:.2f} ms/image")