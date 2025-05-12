import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import antigravity

import base64

@ensure_annotations



def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object.
    Args:
        path_to_yaml (str): Path to the YAML file.
    Returns:
        ConfigBox: A ConfigBox object containing the contents of the YAML file.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except FileNotFoundError as e:
        logger.error(f"File not found: {path_to_yaml}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {path_to_yaml}")
        raise e
    
@ensure_annotations 
def create_directories(path_to_dirs: list, verbose=True):
    """
    Creates directories if they do not exist.
    Args:
        path_to_dirs (list): List of directory paths to create.
        verbose (bool): If True, prints the status of directory creation.
    """
    for dir_path in path_to_dirs:
        os.makedirs(dir_path, exist_ok=True)
        if verbose:
            logger.info(f"Directory {dir_path} created or already exists.")
@ensure_annotations
def save_json(path_to_json: str, data: dict):
    """
    Saves a dictionary as a JSON file.
    Args:
        path_to_json (str): Path to the JSON file.
        data (dict): Dictionary to save.
    """
    with open(path_to_json, "w") as json_file:
        json.dump(data, json_file, indent=4)
        logger.info(f"JSON file {path_to_json} saved successfully.")
@ensure_annotations
def load_json(path_to_json: str) -> dict:
    """
    Loads a JSON file and returns its contents as a dictionary.
    Args:
        path_to_json (str): Path to the JSON file.
    Returns:
        dict: Contents of the JSON file.
    """
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
        logger.info(f"JSON file {path_to_json} loaded successfully.")
        return data
@ensure_annotations
def save_model(path_to_model: str, model):
    """
    Saves a model using joblib.
    Args:
        path_to_model (str): Path to save the model.
        model: The model to save.
    """
    joblib.dump(model, path_to_model)
    logger.info(f"Model saved at {path_to_model}.")
@ensure_annotations
def load_model(path_to_model: str):
    """
    Loads a model using joblib.
    Args:
        path_to_model (str): Path to the model.
    Returns:
        The loaded model.
    """
    model = joblib.load(path_to_model)
    logger.info(f"Model loaded from {path_to_model}.")
    return model
@ensure_annotations     

def encode_image(image_path: str) -> str:       
    """
    Encodes an image to a base64 string.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        logger.info(f"Image {image_path} encoded successfully.")
        return encoded_string
@ensure_annotations
def decode_image(encoded_string: str, output_path: str):
    """
    Decodes a base64 string to an image file.
    Args:
        encoded_string (str): Base64 encoded string of the image.
        output_path (str): Path to save the decoded image.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(encoded_string))
        logger.info(f"Image saved at {output_path}.")
@ensure_annotations
def get_size(path: str) -> int:
    """
    Returns the size of a file or directory.
    Args:
        path (str): Path to the file or directory.
    Returns:
        int: Size in bytes.
    """
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                fp = os.path.join(dirpath, filename)
                total_size += os.path.getsize(fp)
        return total_size
    else:
        raise ValueError(f"{path} is not a valid file or directory.")
@ensure_annotations
def get_file_extension(file_path: str) -> str:
    """
    Returns the file extension of a given file.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: File extension.
    """
    _, file_extension = os.path.splitext(file_path)
    return file_extension
@ensure_annotations
def get_file_name(file_path: str) -> str:
    """
    Returns the file name without extension.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: File name without extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]