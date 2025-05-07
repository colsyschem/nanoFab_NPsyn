import glob
import os, time
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime

# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
folder_path = DATA_UV_DIR_PATH
RESULT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'results', 'figs', '')
REF_SPECTRUM_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'reference_spectrum', '')
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')
OUTPUT_DIR_PATH_2 = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src', '')

current_date = datetime.now().strftime("%Y-%m-%d")

def process_spectrum(df, start_wavelength=300, end_wavelength=1100, increment=1):
    """
    This function processes a given spectral DataFrame by:
    1. Removing the first 14 rows (assumed to be header information).
    2. Normalizing the second column.
    3. Resampling the data to only include wavelengths between 300 and 1100 nm with an increment of 1 nm.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the spectrum data.
    start_wavelength (int, optional): The starting wavelength (default is 300 nm).
    end_wavelength (int, optional): The ending wavelength (default is 1100 nm).
    increment (int, optional): The wavelength increment (default is 1 nm).

    Returns:
    pd.DataFrame: A processed and resampled DataFrame with normalized intensity values.
    """

    # Step 2: Convert the columns to numeric (in case they are not) and handle any conversion errors
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')  # First column assumed to be wavelengths
    df['Absorbance'] = pd.to_numeric(df['Absorbance'], errors='coerce')  # Second column assumed to be intensities

    # Drop any rows with missing data (NaNs)
    df = df.dropna()

    # Step 3: Normalize the second column (intensity or absorbance values)
    df['Absorbance'] = (df['Absorbance'] - df['Absorbance'].min()) / (df['Absorbance'].max() - df['Absorbance'].min())

    # Step 4: Resample the DataFrame to include only wavelengths between start and end, in increments
    new_wavelengths = np.arange(start_wavelength, end_wavelength + increment, increment)

    # Interpolate to fit the new wavelength range
    resampled_df = pd.DataFrame({'Wavelength': new_wavelengths})
    resampled_df['Normalized_Intensity'] = np.interp(new_wavelengths, df.iloc[:,0], df.iloc[:,1])

    return resampled_df

# PLOT THE VARIATION OF LOSS VS ITERATIONS
# Load reference CSV file
#ref_df = pd.read_csv(os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt'), skiprows=14, header=None, delimiter='\t')
#ref_df = process_spectrum(ref_df)

# Initialize empty list to store norm values
norm_values = []
file_names = []

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt') and filename != 'ref_spectrum.txt':
        # Load CSV file into a Pandas DataFrame
        df = pd.read_csv(os.path.join(folder_path, filename), delimiter=',')
        print(df)
        df = process_spectrum(df)

        # Calculate the L2 norm of the difference between the normalized second column of the file and the reference file
        #diff = df.iloc[:, 1] - ref_df.iloc[:, 1]

        #norm_value = np.linalg.norm(diff)
        #norm_value = round(norm_value, 2)
        
        # Add the norm value to the list
        #norm_values.append((filename,norm_value))

# Print the list of all the norm values
#print(natsorted(norm_values))

num_iteration = np.arange(len(norm_values))



# PLOT THE UV-VIS-NIR PRODUCED DURING ITERATIONS
# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.txt'))


# Create a figure with 5 rows and 4 columns of subplots, 12 inches long and  inches wide, 300 dpi
fig, axs = plt.subplots(4, 5, figsize=(6, 6), dpi=300)

# Initialize variables to keep track of subplot indices
row_index = 0
col_index = 0

# Initialize empty list to store Abs at lambda max. / Abs at 400nm
abs_ratios = []
num = 1

# Loop through all CSV files in the folder
for csv_file in natsorted(csv_files):
    
    # Check if the current file is not the reference file
    if os.path.basename(csv_file) != 'resampled_df.csv':
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_file, delimiter=',')
        df = process_spectrum(df)
        
        # Plot the normalized second column of the current file against the reference file in the current subplot
        axs[row_index, col_index].fill_between(df.iloc[:, 0], 0, df.iloc[:, 1], color='firebrick', alpha = 0.5, label='Exp'+str(num))
        num = num+1
        #axs[row_index, col_index].set_xlabel(r'Wavelength (nm)', fontsize='large')
        #axs[row_index, col_index].set_ylabel(r'Norm. Extinction', fontsize='large')
        #axs[row_index, col_index].legend(loc = 'upper left', frameon=False, fontsize='medium')
        axs[row_index, col_index].set_xlim(400,1100)
        axs[row_index, col_index].set_ylim([0.6,1])
        axs[row_index, col_index].spines[['right', 'top']].set_visible(False)

        fig.tight_layout()

        # Increment the column index
        col_index += 1

        # If the column index is equal to 5, reset it to 0 and increment the row index
        if col_index == 5:
            col_index = 0
            row_index += 1

plt.show()
fig.savefig(DATA_UV_DIR_PATH+ f'{current_date}_all_UV-Vis-NIR.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
