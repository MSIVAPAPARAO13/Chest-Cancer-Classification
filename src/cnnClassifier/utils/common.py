import os
import json
import yaml
import joblib
import base64
from pathlib import Path
from typing import Any

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnnClassifier import logger


# =========================
# YAML FUNCTIONS
# =========================

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read YAML file and return as ConfigBox"""

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("YAML file is empty")

            logger.info(f"YAML loaded successfully: {path_to_yaml}")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        logger.exception(f"Error reading YAML: {e}")
        raise e


# =========================
# DIRECTORY FUNCTIONS (FIXED 🔥)
# =========================

def create_directories(paths, verbose: bool = True):
    """Create multiple directories"""

    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


# =========================
# JSON FUNCTIONS
# =========================

@ensure_annotations
def save_json(path: Path, data: dict):
    """Save dictionary as JSON"""

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON and return as ConfigBox"""

    with open(path, "r") as f:
        content = json.load(f)

    logger.info(f"JSON loaded from: {path}")
    return ConfigBox(content)


# =========================
# BINARY FILE FUNCTIONS
# =========================

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save object as binary file"""

    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary file"""

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


# =========================
# FILE SIZE
# =========================

@ensure_annotations
def get_size(path: Path) -> str:
    """Get file size in KB"""

    size_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_kb} KB"


# =========================
# IMAGE UTILITIES
# =========================

def decode_image(img_string: str, file_name: str):
    """Decode base64 string and save as image"""

    img_data = base64.b64decode(img_string)

    with open(file_name, "wb") as f:
        f.write(img_data)

    logger.info(f"Image saved at: {file_name}")


def encode_image_to_base64(image_path: Path) -> bytes:
    """Encode image to base64"""

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read())

    return encoded