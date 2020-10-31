import os
import shutil

import copy
import pickle
import json

import torch

from contactnets.utils import dirs

import pdb

def num_files(path):
    return len(os.listdir(path))

def clear_directory(path):
    """Clear the directory indicated by path, but leave the directory itself."""
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def create_empty_directory(path):
    """Clears a directory if it exists; otherwise, creates a new one."""
    if os.path.exists(path):
        clear_directory(path)
    else:
        os.makedirs(path)

def ensure_created_directory(path):
    """If directory does not exist, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def read_file(path):
    with open(path, 'r') as file:
        return file.read()

def write_file(path, data):
    with open(path, 'w') as file:
        file.write(data)

def stringify_tensors(params):
    for field in params.__dict__:
        attr = getattr(params, field)
        if isinstance(attr, torch.Tensor):
            setattr(params, field, attr.tolist())
    return params

def save_params(params, params_type):
    params = copy.deepcopy(params)

    ensure_created_directory(dirs.out_path('params'))
    with open(dirs.out_path('params', params_type + '.pickle'), 'wb') as file:
        pickle.dump(params, file)
    
    params = stringify_tensors(params)

    with open(dirs.out_path('params', params_type + '.json'), 'w') as file:
        json.dump(params.__dict__, file, sort_keys=True, indent=4)

def load_params(device, params_type):
    # Loads param struct, can be any type
    # Sends tensors to device
    with open(dirs.out_path('params', params_type + '.pickle'), 'rb') as file:
        params = pickle.load(file)

    # Send all tensors to appropriate device
    for field in params.__dict__:
        attr = getattr(params, field)
        if isinstance(attr, torch.Tensor):
            setattr(params, field, attr.to(device))

    return params 
