import os
import shutil
from datetime import datetime

# ======= define paths ===================================

# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')
OUTPUT_DIR_PATH_2 = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src', '')
RESULTS_DIR_PATH = os.path.join(BASE_DIR_PATH, 'results', 'figs', '')

# Define archived files path
ARCHIVED_FILES_PATH = os.path.join(BASE_DIR_PATH, 'archived-files', '')

# ======= define parameters ========================

def move_files_to_archive(source_paths, file_extensions):
    # Get current date in the format YYYY-MM-DD
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the folder name with the format YYYY-MM-DD_archive
    archive_folder_name = f"{current_date}_archive"
    dest_folder = os.path.join(ARCHIVED_FILES_PATH, archive_folder_name)
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Move files with specified extensions from each source path to the destination folder
    for source_path in source_paths:
        if not os.path.exists(source_path):
            print(f"Source path {source_path} does not exist. Skipping.")
            continue

        print(f"Checking files in {source_path}...")  # Debugging line
        # List all files in the source folder
        files = os.listdir(source_path)
        for file_name in files:
            file_path = os.path.join(source_path, file_name)

            # Check extension case-insensitively
            if os.path.isfile(file_path) and file_name.lower().endswith(file_extensions):
                # Move the file to the destination folder
                shutil.move(file_path, dest_folder)
                print(f"Moved {file_name} to {dest_folder}")
            else:
                print(f"Skipping {file_name} (not a valid file type or not a file)")

    print(f"All relevant files moved to {dest_folder}.")

# List of source paths
source_paths = [DATA_UV_DIR_PATH, OUTPUT_DIR_PATH, OUTPUT_DIR_PATH_2, RESULTS_DIR_PATH]

# File extensions to look for
file_extensions = ('.png', '.jpg', '.jpeg', '.pdf', '.txt', '.csv')

# Run the function to move files to the archive
move_files_to_archive(source_paths, file_extensions)
