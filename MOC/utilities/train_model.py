import os
# For exact CUDA error lines: restart the kernel, uncomment this, then run from the top.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchlogix.layers import LogicConv2d, LogicDense, OrPooling2d, GroupSum

def train_model(
    model,
    train_dataset,
    test_dataset,
    lr=2e-1,
    weight_decay=0,
    batch_size=512,
    num_iterations=2500,
    metrics_every=100,
    force_cpu=False,
):
    device = torch.device(
        "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    def load_n(loader, n):
        i = 0
        while i < n:
            for batch in loader:
                yield batch
                i += 1
                if i == n:
                    return

    def evaluate(loader, train_mode=False):
        orig_mode = model.training
        model.train(train_mode)

        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += loss_fn(logits, y).item() * y.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        model.train(orig_mode)
        return total_loss / total, total_correct / total

    running_loss = 0.0
    running_examples = 0

    test_loss_acc = {
        "train_iteration": [],
        "test_acc_discrete": [],
        "test_loss_discrete": [],
        "test_acc_relaxed": [],
        "test_loss_relaxed": [],
    }

    for step, (x, y) in enumerate(load_n(train_loader, num_iterations), start=1):
        model.train()
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_examples += y.size(0)

        if step % metrics_every == 0:
            disc_loss, disc_acc = evaluate(test_loader, train_mode=False)
            relax_loss, relax_acc = evaluate(test_loader, train_mode=True)
            train_loss = running_loss / running_examples

            test_loss_acc["train_iteration"].append(step)
            test_loss_acc["test_acc_discrete"].append(disc_acc)
            test_loss_acc["test_loss_discrete"].append(disc_loss)
            test_loss_acc["test_acc_relaxed"].append(relax_acc)
            test_loss_acc["test_loss_relaxed"].append(relax_loss)
            match save:
              case "None":
                  pass

              case "best":
                  if disc_acc > best_acc:
                      best_acc = disc_acc
                      torch.save(model.state_dict(), "logicnet_bloodmnist_best.pth")
                      

              case "last":
                  torch.save(model.state_dict(), "logicnet_bloodmnist_last.pth")
                  

              case _:
                  raise ValueError(f"Unknown save mode: {save}")
            print(
                f"iter {step:4d} | "
                f"train_loss {train_loss:.4f} | "
                f"test_acc_discrete {disc_acc:.4f} | test_loss_discrete {disc_loss:.4f} | "
                f"test_acc_relaxed {relax_acc:.4f} | test_loss_relaxed {relax_loss:.4f}"
            )

            running_loss = 0.0
            running_examples = 0

    return model, test_loss_acc
