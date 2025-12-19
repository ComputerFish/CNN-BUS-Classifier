"""
This module performs K-fold cross-validation training, validation, and testing
of a CNN image classification model using PyTorch. It manages data loading,
training loops, model evaluation, and logging of per-fold metrics.

This is the main file that runs all the other modules, you can either run it
on your personal computer in the CLI:
    python main.py [arguments if needed]

or on a SLURM cluster by running the slurm_run.sh file on the CLI:
    sbatch slurm_run.sh

The training pipeline includes:
    - Reproducible seeding
    - K-fold dataset splitting
    - Model training and validation per epoch
    - Metrics logging and saving for each fold
    - Aggregation of fold-level test results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
import torchvision
from tqdm import tqdm

from dataset import CSVDataset
from model import CNNModel
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any

def run_training(args: Any) -> None:
    """
    Execute K-fold cross-validation training and evaluation.

    Args:
        args: Object containing configuration parameters such as dataset directories and image settings.        
    Returns:
        None. Results and logs are written to disk.
    """
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(2.0)

    # Load dataset
    csv_file_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(csv_file_path)
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)

    # Map tumor type to binary label
    df["tumor"] = df["tumor_type"].map({"Benign": 0, "Malignant": 1})

    # KFold split
       # 
    # ---------------------------------------------------------------
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    dfSplit = kf.split(df)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(dfSplit):
        print(f"Starting fold {fold + 1}/{args.num_folds}...")

        train_df = df.iloc[train_idx]
        # val_df = df.iloc[test_idx]
        test_df = df.iloc[test_idx]

        # Further split training data into train/validation 80/20 split of the training data
        train_subset, val_subset = train_test_split(
            train_df, 
            test_size=0.2, 
            random_state=args.seed, 
            shuffle=True
        )

    # --------------------------------------------------------------

        # Initialize datasets and dataloaders
        train_ds = CSVDataset(args, train_subset)
        val_ds = CSVDataset(args, val_subset)
        test_ds = CSVDataset(args, test_df)

        num_workers = os.cpu_count()-1

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers, persistent_workers=True)
        # Initialize model, loss, and optimizer
        model = CNNModel(args).to(DEVICE)
        # ------------- Regular BCE loss -------------
        criterion = torch.nn.BCELoss()

        # ------------- Weighted BCE loss -------------
        # numOfPos = len(train_subset[train_subset['tumor'] == 1])
        # numOfNeg = len(train_subset[train_subset['tumor'] == 0])
        # w_pos = numOfNeg / (numOfPos + numOfNeg)
        # w_neg = numOfPos / (numOfPos + numOfNeg)
        # weights=torch.tensor([w_neg, w_pos]).to(DEVICE)
        # criterion = torch.nn.BCELoss(weight=weights)

        # ------------- Focal Loss -------------
        # numOfPos = len(train_subset[train_subset['tumor'] == 1])
        # numOfNeg = len(train_subset[train_subset['tumor'] == 0])
        # w_pos = numOfNeg / (numOfPos + numOfNeg)
        # w_neg = numOfPos / (numOfPos + numOfNeg)
        # def focal_loss_wrapper(inputs, targets):
        #     return torchvision.ops.sigmoid_focal_loss(
        #         inputs=inputs, targets=targets, alpha=w_pos, gamma=2.0, reduction="mean"
        #     )
        # criterion = focal_loss_wrapper
        # ---------------------------------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        log_records = []

        # Training loop
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_roc, val_pr = validate(
                model, val_dl, criterion, DEVICE
            )

            log_records.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_f1": val_f1,
                "val_roc_auc": val_roc,
                "val_pr_auc": val_pr,
            })

            print(
                f"Epoch [{epoch + 1:02d}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )

        # Save per-fold logs
        log_df = pd.DataFrame(log_records)
        log_path = os.path.join(args.output_dir, args.version, "logs", f"fold_{fold + 1}_log.csv")
        log_df.to_csv(log_path, index=False)

        # Final test evaluation
        test_acc, test_prec, test_rec, test_f1, test_roc, test_pr = test(model, test_dl, DEVICE)
        fold_results.append({
            "fold": fold + 1,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc,
            "test_pr_auc": test_pr,
        })

        # Free memory and delete model to prevent leakage over folds
        del model
        torch.cuda.empty_cache()

    # Save aggregated test results
    results_path = os.path.join(args.output_dir, args.version, "testing_results.csv")
    pd.DataFrame(fold_results).to_csv(results_path, index=False)


if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
