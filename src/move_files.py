import os
import shutil
from datetime import datetime

# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
SPECTROPHOTOMER_INITIALIZATION_DATA_PATH = os.path.join(BASE_DIR_PATH, 'src', '')
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
REF_SPECTRUM_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'reference_spectrum', '')
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')
OUTPUT_DIR_PATH_2 = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src', '')
RESULTS_DIR_PATH = os.path.join(BASE_DIR_PATH, 'results', 'figs', '')

def move_files_to_experiment_folder(dest_path, experiment_name, source_paths, file_extensions):
    # Get current date in the format YYYY-MM-DD
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the folder name in the format YYYY-MM-DD_experiment-name
    folder_name = f"{current_date}_{experiment_name}"
    dest_folder = os.path.join(dest_path, folder_name)
    
    # Create the destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Move only files with specified extensions from each source path to the destination folder
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

# Example usage:
experiment_name = input('Type the name of the experiment: ')

# Check and print the files in the results directory to verify .png is present
print(f"Files in {RESULTS_DIR_PATH}: {os.listdir(RESULTS_DIR_PATH)}")

# Move .txt and .csv files to a new folder in DATA_UV_DIR_PATH
move_files_to_experiment_folder(DATA_UV_DIR_PATH, experiment_name, [OUTPUT_DIR_PATH, OUTPUT_DIR_PATH_2, DATA_UV_DIR_PATH, REF_SPECTRUM_DIR_PATH], ('.csv', '.txt'))

# Move .jpg, .png, .pdf files to a new folder in RESULTS_DIR_PATH
move_files_to_experiment_folder(RESULTS_DIR_PATH, experiment_name, [DATA_UV_DIR_PATH], ('.jpg', '.png', '.pdf'))

# Move .txt file from src folder having spectrophotometer-initialization-data to DATA_UV_DIR_PATH
move_files_to_experiment_folder(DATA_UV_DIR_PATH, experiment_name, [SPECTROPHOTOMER_INITIALIZATION_DATA_PATH], ('.csv', '.txt'))
