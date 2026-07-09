import io

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    return total_loss / max(1, len(dataloader)), acc, f1m, cm_norm, (y_true, y_pred)


def text_confusion_matrix(cm, labels=None, width=7):
    if labels is None:
        labels = [f"C{i}" for i in range(cm.shape[0])]

    out = io.StringIO()
    header = "Pred↓ / True→ | " + "  ".join([f"{label:>{width}}" for label in labels])
    print(header, file=out)
    print("-" * len(header), file=out)

    for i in range(cm.shape[0]):
        row = "  ".join([f"{cm[i, j]:>{width}.2f}" for j in range(cm.shape[1])])
        print(f"{labels[i]:>16} | {row}", file=out)

    return out.getvalue()


class EarlyStopping:
    def __init__(self, patience=5, mode="min", min_delta=0.0):
        if mode not in ("min", "max"):
            raise ValueError("mode debe ser 'min' o 'max'.")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        self.best_state = None
        self.best_epoch = 0

    def _is_better(self, current, best):
        if best is None:
            return True

        if self.mode == "min":
            return (best - current) > self.min_delta

        return (current - best) > self.min_delta

    def step(self, metric_value, model, epoch_idx):
        if self.best is None or self._is_better(metric_value, self.best):
            self.best = metric_value
            self.num_bad = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch_idx
            return False

        self.num_bad += 1
        return self.num_bad > self.patience

    def load_best_weights(self, model, device):
        if self.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})