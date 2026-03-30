from pathlib import Path
from typing import Any, Dict
import yaml
import os

from dotenv import load_dotenv

#load_dotenv()  # <-- this is required

#BASE_DIR = Path(os.environ["PROJECT_ROOT"])

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_data_config() -> Dict[str, Any]:
    return load_config(Path("configs/data.yaml"))


def load_model_config() -> Dict[str, Any]:
    return load_config(Path("configs/model.yaml"))


def load_train_config() -> Dict[str, Any]:
    return load_config(Path("configs/train.yaml"))