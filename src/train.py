"""Training loop with validation tracking, L2 (weight decay), LR scheduling and early stopping.

Run from the project root, e.g. on Kaggle:
    python -m src.train --model resnet18 --mode fine_tune --epochs 8 --lr 1e-4
"""
import argparse
import copy
import time

import torch
import torch.nn as nn

from src import config
from src.models import build_model, save_checkpoint
from src.preprocessing import get_class_names, get_dataloaders, load_oct_dataset
from src.utils import count_parameters, get_device, plot_history, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += images.size(0)
    return total_loss / seen, correct / seen


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3,
                weight_decay=1e-4, patience=4):
    """Train with Adam (+ L2 via weight_decay), ReduceLROnPlateau and early stopping.

    Returns (best_model, history). 'Best' = lowest validation loss.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, wait = float("inf"), None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_loss:
            best_loss, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history["train_time_sec"] = time.time() - start
    model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=config.MODEL_NAMES)
    parser.add_argument("--mode", default="fine_tune",
                        choices=["feature_extraction", "fine_tune", "full"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_per_class", type=int, default=config.TRAIN_PER_CLASS)
    parser.add_argument("--tag", default="", help="suffix for output file names")
    parser.add_argument("--no_batchnorm", action="store_true", help="Disable BatchNorm in CustomCNN")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("Device:", device)

    dataset = load_oct_dataset()
    class_names = get_class_names(dataset)
    pretrained = args.model != "custom_cnn"
    train_loader, val_loader, _ = get_dataloaders(
        dataset, pretrained=pretrained, batch_size=args.batch_size,
        train_per_class=args.train_per_class)

    model = build_model(args.model, len(class_names), mode=args.mode, dropout=args.dropout, use_batchnorm=not args.no_batchnorm)
    print(f"Trainable parameters: {count_parameters(model):,}")

    model, history = train_model(model, train_loader, val_loader, device,
                                 epochs=args.epochs, lr=args.lr,
                                 weight_decay=args.weight_decay)

    run_name = f"{args.model}_{args.mode if pretrained else 'scratch'}{args.tag}"
    save_checkpoint(model, args.model, config.MODEL_DIR / f"{run_name}.pt", class_names,
                    extra={"mode": args.mode, "lr": args.lr, "use_batchnorm": not args.no_batchnorm})
    history["trainable_params"] = count_parameters(model)
    save_json(history, config.RESULT_DIR / f"{run_name}_history.json")
    plot_history(history, run_name, config.FIGURE_DIR / f"{run_name}_curves.png")
    print(f"Saved model + curves for {run_name}")


if __name__ == "__main__":
    main()
