import os
import zipfile
from pathlib import Path


def unzip_all_in_directory(root_dir):
    """
    Recursively unzip all ZIP files in the specified directory and its subdirectories.

    Args:
        root_dir (str): The root directory to search for ZIP files.
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    for zip_file in root_path.rglob('*.zip'):
        try:
            # Create output directory with the same name as the ZIP file (without extension)
            output_dir = zip_file.parent / zip_file.stem
            output_dir.mkdir(exist_ok=True)

            # Extract the ZIP file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            print(f"Successfully extracted: {zip_file} to {output_dir}")

        except Exception as e:
            print(f"Failed to extract {zip_file}: {str(e)}")


if __name__ == "__main__":
    # Specify the root directory to search for ZIP files
    root_directory = '/Volumes/Zhengfei_01/Fluxnet2025'

    # Call the function to unzip all files
    unzip_all_in_directory(root_directory)
