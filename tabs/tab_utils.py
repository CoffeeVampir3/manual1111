import os, sys
import glob
import torch

def find_leaf_directories(parent_directory):
    leaf_directories = []
    for item in os.scandir(parent_directory):
        # If it's a directory, check if it has any visible subdirectories
        if item.is_dir():
            leaf_directories.append(item.path)
    return leaf_directories

def find_leaf_files(parent_directory, valid_extensions=['*']):
    valid_extensions = [ext if ext.startswith('.') else '.' + ext for ext in valid_extensions]
    files = []
    for ext in valid_extensions:
        files.extend(glob.glob(os.path.join(parent_directory, '*' + ext)))
    return files

def get_available_from_dir(target):
    return [os.path.basename(x) for x in find_leaf_directories(target)]

def get_available_from_leafs(target, valid_extensions=['*']):
    return [os.path.basename(x) for x in find_leaf_files(target, valid_extensions)]

def get_available_devices():
    cuda_devices = "Use only CUDA Devices: "
    if torch.cuda.is_available():  # If CUDA devices are available
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            cuda_devices += f'{device_name} (cuda:{i}) '
    return ["Use only CPU", cuda_devices]