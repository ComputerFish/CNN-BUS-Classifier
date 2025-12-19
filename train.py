"""
This module provides utility functions for training, validating, and testing
a PyTorch classification model. It includes per-epoch training, validation with
metric computation, and final model evaluation.

Functions:
    train_one_epoch: Train the model for a single epoch.
    validate: Evaluate model performance on the validation set.
    test: Evaluate model performance on the test set.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from metrics import calculate_metrics
from typing import Tuple


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch on the provided dataset.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average training loss and accuracy for the epoch.
    """
    model.train()
    total_train_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in dataloader:
        # Move input data and labels to the target device
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * imgs.size(0)

        pred_classes = torch.argmax(outputs, dim=1)
        true_classes = torch.argmax(labels, dim=1)
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(true_classes.cpu().numpy())

    avg_loss = total_train_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Validate the model on the provided dataset and compute detailed metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float, float]:
            Average validation loss, accuracy, precision, recall,
            F1 score, ROC AUC, and PR AUC.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(true_classes.cpu().numpy())

            probs = torch.max(outputs, dim=1).values
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc, prec, rec, f1, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)
    return avg_loss, acc, prec, rec, f1, roc_auc, pr_auc


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Test the trained model on the test dataset and compute performance metrics.

    Args:
        model (nn.Module): Trained model to evaluate.
        dataloader (DataLoader): DataLoader providing test data.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Accuracy, precision, recall, F1 score, ROC AUC, and PR AUC.
    """
    # 
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(imgs)
            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(true_classes.cpu().numpy())

            probs = torch.max(outputs, dim=1).values
            all_probs.extend(probs.cpu().numpy())

    acc, prec, rec, f1, roc_auc, pr_auc = calculate_metrics(all_labels, all_preds, all_probs)
    return acc, prec, rec, f1, roc_auc, pr_auc
