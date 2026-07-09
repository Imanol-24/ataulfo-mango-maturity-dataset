import argparse

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import build_model
from utils import eval_one_epoch, text_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mobilenetv2", "mobilenetv3", "resnet18"], required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    criterion = nn.CrossEntropyLoss()
    val_loss, acc, f1m, cm_norm, _ = eval_one_epoch(model, val_loader, criterion, device)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1m:.4f}")
    print("\nMatriz de confusión normalizada (filas = clase real):")
    print(text_confusion_matrix(cm_norm, labels=classes))


if __name__ == "__main__":
    main()