import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model
from utils import eval_one_epoch, train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mobilenetv2", "mobilenetv3", "resnet18"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} | "
            f"Val Acc = {val_acc:.4f} | "
            f"Val F1 = {val_f1:.4f}"
        )

    output_path = f"{args.model}_mango_model.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Modelo guardado en: {output_path}")


if __name__ == "__main__":
    main()