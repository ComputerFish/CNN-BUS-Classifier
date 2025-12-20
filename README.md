# CNN BUS Classifier

A beginner-friendly PyTorch project for building and training Convolutional Neural Network (CNN) models for binary image classification. This project specifically focuses on classifying breast ultrasound (BUS) images as **Benign** or **Malignant**.

## 🎯 Project Overview

This repository is designed as a learning project to understand how to:
- Build CNN architectures from scratch using PyTorch
- Implement a complete training pipeline with K-fold cross-validation
- Handle medical image datasets
- Track and evaluate model performance with multiple metrics
- Deploy models on both local machines and HPC clusters

The project implements a custom CNN model trained to classify breast ultrasound images, achieving binary classification between benign and malignant tumors.

## Metrics Across models
<table align="center">
  <tr>
    <td>
      <img width="553" height="254" alt="Screenshot 2025-12-19 at 1 17 17 PM" src="https://github.com/user-attachments/assets/03aae67a-0d29-400b-a579-1ea80b9f22ca" />
    </td>
  </tr>
</table>

## 🏗️ Architecture

The CNN model consists of:

**Current Architecture (Model 3 - Custom):**
- 4 Convolutional blocks with the following pattern:
  - **Block 1:** Conv2D(3→32, kernel=5×5) → BatchNorm → ReLU → MaxPool(3×3, stride=2)
  - **Block 2:** Conv2D(32→64, kernel=3×3) → BatchNorm → ReLU → MaxPool(3×3, stride=2)
  - **Block 3:** Conv2D(64→128, kernel=5×5) → BatchNorm → ReLU → MaxPool(3×3, stride=2)
  - **Block 4:** Conv2D(128→256, kernel=3×3) → BatchNorm → ReLU → MaxPool(3×3, stride=2)
- Global Average Pooling layer
- Fully connected output layer (256→2)
- Sigmoid activation for binary classification

> **Note:** The `model.py` file includes commented-out alternative architectures that you can experiment with for learning purposes.

## 📁 Dataset Structure

The project expects the following dataset structure:

```
Dataset/
├── dataset.csv
└── images/
    ├── case_001/
    │   └── image.png
    ├── case_002/
    │   └── image.png
    └── ...
```

**CSV Format:**
The `dataset.csv` file should contain at least these columns:
- `case_id`: Directory name containing the image (e.g., "case_001")
- `tumor_type`: Label as either "Benign" or "Malignant"

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support for GPU training, or MPS for Apple Silicon)
- Required Python packages

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ComputerFish/CNN-BUS-Classifier.git
cd CNN-BUS-Classifier
```

2. Install dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow tqdm
```

3. Prepare your dataset in the structure shown above.

## 💻 Usage

### Local Training (CLI)

Run the training script with default parameters:

```bash
python main.py
```

Or customize the training parameters:

```bash
python main.py --epochs 30 --batch 64 --lr 1e-4 --version MyExperiment_v1
```

### SLURM Cluster Training

For training on an HPC cluster with SLURM:

1. Edit `slurm_run.sh` with your cluster settings:
   - Update `--nodelist` with your target node
   - Update conda environment path and name
   - Modify resource requirements as needed

2. Submit the job:
```bash
sbatch slurm_run.sh
```

## ⚙️ Configuration Options

The training pipeline is highly configurable via command-line arguments:

### Dataset Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | `./Dataset/` | Root directory of the dataset |
| `--image_dir` | `images` | Directory containing image folders |
| `--csv_file` | `dataset.csv` | CSV file with labels |
| `--image_file` | `image.png` | Image filename in each case folder |

### Preprocessing Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | `42` | Random seed for reproducibility |
| `--image_size` | `160` | Resize images to this dimension (160×160) |
| `--channel` | `3` | Number of image channels (3 for RGB) |
| `--norm_mean` | `0.5` | Mean for normalization |
| `--norm_std` | `0.5` | Standard deviation for normalization |
| `--num_classes` | `2` | Number of output classes |

### Training Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `20` | Number of training epochs |
| `--num_folds` | `5` | Number of K-fold splits |
| `--batch` | `32` | Batch size |
| `--lr` | `1e-5` | Learning rate |
| `--output_dir` | `./outputs` | Directory for saving results |
| `--device_name` | `0` | GPU device ID |
| `--version` | `Model_1` | Experiment version name |

### Example Commands

Train with custom learning rate and batch size:
```bash
python main.py --lr 1e-4 --batch 64 --epochs 50
```

Train with a specific experiment version:
```bash
python main.py --version CustomCNN_v2 --num_folds 10
```

## 📊 Outputs

The training process generates the following outputs in `outputs/<version>/`:

```
outputs/
└── Model_1/
    ├── logs/
    │   ├── fold_1_log.csv
    │   ├── fold_2_log.csv
    │   ├── ...
    │   └── fold_5_log.csv
    └── testing_results.csv
```

### Log Files

**Per-fold logs** (`fold_X_log.csv`) contain epoch-by-epoch metrics:
- `fold`: Fold number
- `epoch`: Epoch number
- `train_loss`: Training loss
- `train_acc`: Training accuracy
- `val_loss`: Validation loss
- `val_acc`: Validation accuracy
- `val_prec`: Validation precision
- `val_rec`: Validation recall
- `val_f1`: Validation F1 score
- `val_roc_auc`: Validation ROC AUC
- `val_pr_auc`: Validation PR AUC

**Testing results** (`testing_results.csv`) contain final test metrics per fold:
- `fold`: Fold number
- `test_acc`: Test accuracy
- `test_prec`: Test precision
- `test_rec`: Test recall
- `test_f1`: Test F1 score
- `test_roc_auc`: Test ROC AUC
- `test_pr_auc`: Test PR AUC

## 📂 Project Structure

```
CNN-BUS-Classifier/
├── main.py           # Main training orchestration with K-fold CV
├── model.py          # CNN architecture definitions
├── dataset.py        # Custom PyTorch Dataset class
├── train.py          # Training, validation, and testing functions
├── metrics.py        # Performance metrics calculation
├── config.py         # Command-line argument configuration
├── slurm_run.sh      # SLURM job submission script
└── README.md         # This file
```

### Module Descriptions

- **`main.py`**: Orchestrates the entire training pipeline, including K-fold cross-validation, data splitting, and result aggregation.
- **`model.py`**: Defines the CNN architecture. Contains multiple model variants (commented out) for experimentation.
- **`dataset.py`**: Implements a custom PyTorch Dataset that loads images and labels from CSV and applies transformations.
- **`train.py`**: Contains functions for training one epoch, validating the model, and testing the final model.
- **`metrics.py`**: Computes classification metrics (accuracy, precision, recall, F1, ROC AUC, PR AUC).
- **`config.py`**: Defines all configurable hyperparameters using argparse.

## 🎓 Learning Points

This project demonstrates:

1. **CNN Architecture Design**: Building a custom CNN with convolutional layers, batch normalization, and pooling
2. **PyTorch Fundamentals**: Creating custom datasets, data loaders, and training loops
3. **Cross-Validation**: Implementing K-fold CV for robust model evaluation
4. **Medical Image Classification**: Handling real-world medical imaging data
5. **Performance Metrics**: Computing and tracking multiple classification metrics
6. **Experiment Management**: Organizing outputs and tracking different model versions
7. **HPC Integration**: Running ML experiments on cluster environments

## 🔧 Alternative Loss Functions

The code includes commented examples of alternative loss functions you can experiment with:

- **Regular BCE Loss** (currently active)
- **Weighted BCE Loss**: For handling class imbalance
- **Focal Loss**: For focusing on hard-to-classify examples

To switch loss functions, uncomment the desired loss function block in `main.py` (lines 102-123).

## 🖥️ Device Support

The model automatically detects and uses available hardware:
- **CUDA**: For NVIDIA GPUs
- **MPS**: For Apple Silicon (M1/M2) GPUs
- **CPU**: Fallback if no GPU is available
