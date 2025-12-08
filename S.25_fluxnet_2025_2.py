import os
import shutil
from pathlib import Path


def remove_non_archive_folders(root_dir):
    """
    Remove all subfolders in the specified directory that don't contain 'ARCHIVE' in their names.

    Args:
        root_dir (str): The root directory to search for subfolders.
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        return

    # Get all subdirectories
    for folder in root_path.iterdir():
        if folder.is_dir() and 'ARCHIVE' not in folder.name:
            try:
                shutil.rmtree(folder)
                print(f"Removed: {folder}")
            except Exception as e:
                print(f"Failed to remove {folder}: {str(e)}")


if __name__ == "__main__":
    # Specify the root directory to process
    target_directory = '/Volumes/Zhengfei_01/Fluxnet2025/ICOS'

    # Call the function to remove non-ARCHIVE folders
    remove_non_archive_folders(target_directory)
