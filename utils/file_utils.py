# file_utils.py

import os

def save_file(file, save_path):
    """Save an uploaded file."""
    if file:
        file.save(save_path)

def remove_file_if_exists(file_path):
    """Remove a file if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)

def get_full_path(base_dir, *subdirs):
    """Generate a full path given a base directory and subdirectories."""
    return os.path.join(base_dir, *subdirs)
