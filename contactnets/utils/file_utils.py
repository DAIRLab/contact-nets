import copy
import json
import os
import pdb  # noqa
import pickle
import shutil
from typing import Any

import torch

from contactnets.utils import dirs


def num_files(path: str) -> int:
    """Number of files in the specified directory."""
    return len(os.listdir(path))


def clear_directory(path: str) -> None:
    """Clear the directory indicated by path, but leave the directory itself."""
    for root, directories, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in directories:
            shutil.rmtree(os.path.join(root, d))


def create_empty_directory(path: str) -> None:
    """Clear a directory if it exists; otherwise, creates a new one."""
    if os.path.exists(path):
        clear_directory(path)
    else:
        os.makedirs(path)


def ensure_created_directory(path: str) -> None:
    """If directory does not exist, create it."""
    if not os.path.exists(path):
        os.makedirs(path)


def read_file(path: str) -> str:
    """Read a text file."""
    with open(path, 'r') as file:
        return file.read()
    raise Exception("Couldn't read file")


def write_file(path: str, data: str) -> None:
    """Write text to a file."""
    with open(path, 'w') as file:
        file.write(data)


def listify_tensors(params: Any) -> Any:
    """Iterate over an object dictionary, turning any tensors into list representations."""
    for field in params.__dict__:
        attr = getattr(params, field)
        if isinstance(attr, torch.Tensor):
            setattr(params, field, attr.tolist())
    return params


def save_params(params: Any, params_type: str) -> None:
    """Write a parameter object in pickle and json representations.

    Parameters will be written to the 'params' directory.

    Args:
        params: the object to serialize.
        params_type: the prefix of the serialized files.
    """
    params = copy.deepcopy(params)

    ensure_created_directory(dirs.out_path('params'))
    with open(dirs.out_path('params', params_type + '.pickle'), 'wb') as file_pickle:
        pickle.dump(params, file_pickle)

    params = listify_tensors(params)

    with open(dirs.out_path('params', params_type + '.json'), 'w') as file_json:
        json.dump(params.__dict__, file_json, sort_keys=True,
                  indent=4, default=lambda x: x.__dict__)


def load_params(device: torch.device, params_type: str) -> Any:
    """Read a parameter object, moving any tensors to the specified device.

    Parameters will be read from the 'params' directory.

    Args:
        device: the device to move tensors to.
        params_type: the prefix of the serialized files.
    """

    with open(dirs.out_path('params', params_type + '.pickle'), 'rb') as file:
        params = pickle.load(file)

    for field in params.__dict__:
        attr = getattr(params, field)
        if isinstance(attr, torch.Tensor):
            setattr(params, field, attr.to(device))

    return params
