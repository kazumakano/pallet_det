import json
import os.path as path
from typing import Any
import torch
import yaml


def load_param(param_file: str) -> dict[str, Any]:
    with open(param_file) as f:
        match path.splitext(param_file)[1]:
            case ".json":
                return json.load(f)
            case ".yaml":
                return yaml.safe_load(f)
            case _:
                raise Exception("only json and yaml are supported")

# same as seclage
def random_split(files: list[str], prop: tuple[float, float, float], seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    mixed_idxes = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed), dtype=torch.int32).numpy()

    train_num = round(prop[0] * len(mixed_idxes) / sum(prop))
    train_files = []
    for i in mixed_idxes[:train_num]:
        train_files.append(files[i])

    val_num = round(prop[1] * len(mixed_idxes) / sum(prop))
    val_files = []
    for i in mixed_idxes[train_num:train_num + val_num]:
        val_files.append(files[i])

    test_files = []
    for i in mixed_idxes[train_num + val_num:]:
        test_files.append(files[i])

    return train_files, val_files, test_files
