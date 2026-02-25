import torch
from torch.utils.data import DataLoader
import numpy as np


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    model.to(device)
    model.eval()

    batch_acc = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = torch.argmax(model(inputs), 1)

            correct = predictions == targets
            
            for i in range(len(predictions)):
                prediction = int(predictions[i].cpu())
                target = int(targets[i].cpu())
                
                all_predictions.append(prediction)
                all_targets.append(target)
            
            accuracy = correct.float().mean().item()

            batch_acc.append(accuracy)

    avg_accuracy = np.mean(batch_acc)

    return avg_accuracy, all_predictions, all_targets
