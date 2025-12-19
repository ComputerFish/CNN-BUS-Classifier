"""
This module defines hyperparameters and command-line arguments for model training.

It uses Python's `argparse.ArgumentParser` to provide a structured interface for 
configuring runtime parameters such as training epochs, device_name, and 
other customizable options. These arguments are parsed from the command line and 
used throughout the training pipeline.

Example:
    To train a model with 32 epochs and specify an available device:
        python main.py --epochs 32 --device_name 2
"""

import argparse

config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type = str, default = "./Dataset/", help = "The root directory of the dataset")
config_args.add_argument('--image_dir', type = str, default = "images", help = "The root directory of the images files")
config_args.add_argument('--csv_file', type = str, default = "dataset.csv", help = "CSV file name")
config_args.add_argument('--image_file', type = str, default = "image.png", help = "Image file name")

#----------------------------------- Preprocessing Configs -----------------------------------#
config_args.add_argument('--seed', type = int, default = 42, help = "seed for reproduciability")
config_args.add_argument('--image_size', type = int, default = 160, help = "Resize images to dimention image_size X image_size")
config_args.add_argument('--channel', type = int, default = 3, help = "Number of channels of the input image, 1 for grayscale and 3 for RGB.")
config_args.add_argument('--norm_mean', type = float, default = 0.5, help = "Mean value for normalization")
config_args.add_argument('--norm_std', type = float, default = 0.5, help = "STD value for normalization")
config_args.add_argument('--num_classes', type = int, default = 2, help = "# of classes")

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--epochs', type = int, default = 20, help = "# of epochs")
config_args.add_argument('--num_folds', type = int, default = 5, help = "# of folds")
config_args.add_argument('--batch', type = int, default = 32, help = "batch size")
config_args.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")
config_args.add_argument('--lr', type = float, default = 1e-5, help = "learning rate")
config_args.add_argument('--output_dir', type = str, default = "./outputs", help = "The root directory of the outputs")
config_args.add_argument('--device_name', type = str, default = "0", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--version', type = str, default = "Model_1", help = "The name of the version run (creates a directory based on the name).")
