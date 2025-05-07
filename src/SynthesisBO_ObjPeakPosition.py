import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.factory import Models
import seabreeze.spectrometers as sb
from pathlib import Path
from tqdm import tqdm
import time
from scipy.interpolate import interp1d

print('Successfully imported all the packages.')

# =============== Define paths =============================
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
folder_path = DATA_UV_DIR_PATH
AX_TO_PUMP_1_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')
AX_TO_PUMP_2_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src', '')
# Define the text file path for saving suggested parameters for PUMP injection action
parameters2pump_1_path = os.path.join(AX_TO_PUMP_1_DIR_PATH, 'suggested_parameters.txt')
parameters2pump_2_path = os.path.join(AX_TO_PUMP_2_DIR_PATH, 'suggested_parameters_2.txt')


# ================ Objective functions ================================
obj1_name='peak_position_diff' # Difference between objective peak position and obtained from the iteration
obj2_name='intensity_ratio'

# ================ Define parameters ================

obj_peak_position = 800 #nm
num_iterations = 15
num_trials_SOBOL = 2
seed_SOBOL = 103 # This number has no physical significance 

# Connect to the spectrophotometer
devices = sb.list_devices()
if not devices:
    print('No Spectrophotometer Found!')
spec = sb.Spectrometer(devices[0])
print('Spectrophotometer successfully connected.')
spec.integration_time_micros(40000)

# Initialize lists to store data for plotting
peak_position_values = []  # Updated name
intensity_ratio_values = []
iteration_numbers = []
parameter_history = []
Net_volume = []         #Total amont of liquid volume transfere through the cleaning channel

# Load baseline data
baseline_data = pd.read_csv('spectrometer_data.txt', delimiter = '\t')

# After sending an order to a pump wait indicated seconds
def sleep_with_progress(seconds, description="Waiting"):
    """Display a progress bar while sleeping for a given number of seconds."""
    for _ in tqdm(range(seconds), desc=description, ncols=80):
        time.sleep(1)

# Define the updated objective function
def objective_function(processed_data, obj1_name='peak_position_diff', obj2_name='intensity_ratio'):
    # Objective 1
    # Find peak position (wavelength of maximum)
    data_500_1100 = processed_data[(processed_data["Wavelength"] >= 500) & (processed_data["Wavelength"] <= 1100)] # Crop spectrum 500 to 1100 nm
    peak_row = data_500_1100.loc[data_500_1100["Intensity"].idxmax()] # Wavelength and intensity of the maximum in 500-1100 range
    peak_position_value = peak_row["Wavelength"] # Wavelength of maximum
    peak_position_diff = obj_peak_position - peak_position_value # Obj1: Difference between objective peak position and obtained from the iteration

    # Objective 2
    # Peak intensity at objective wavelength
    intensity_at_objective_wavelength = processed_data.loc[processed_data["Wavelength"]  == obj_peak_position, "Intensity"]
    intensity_at_objective_wavelength = intensity_at_objective_wavelength.values[0] if not intensity_at_objective_wavelength.empty else 0
    # Find intensity at 400 nm
    intensity_at_400 = processed_data.loc[processed_data["Wavelength"]  == 400, "Intensity"]
    intensity_at_400 = intensity_at_400.values[0] if not intensity_at_400.empty else 0
    # Calculate intensity ratio
    ratio = intensity_at_objective_wavelength / intensity_at_400 if intensity_at_400 != 0 else np.inf #Obj2: Ratio between intensity of objective wavelength and instensity at 400 nm

    # Calculate the amount of gold used
    Au_inmM = (((0.5)/(0.12))* intensity_at_400) # Since absorbance of 0.12 (for 1mm optical path) corresponds to 0.5 mM Au

    print(f"Calculated peak position diff: {peak_position_diff} nm")
    print(f"Calculated intensity ratio: {ratio}")
    print(f'Calculated amount of Au: {Au_inmM} mM')

    return {obj1_name: peak_position_diff, obj2_name: ratio}

def process_spectrum(df, start_wavelength=400, end_wavelength=1100, increment=1):
    # Assign column names explicitly for wavelength and intensity
    df.columns = ['Wavelength', 'Intensity']  # Adjust if your data has more columns or different structure

    # Convert columns to numeric, handling any errors
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')

    # Drop any rows with missing data (NaNs)
    df = df.dropna()

    # Resample the DataFrame to include only wavelengths between start and end, in increments
    new_wavelengths = np.arange(start_wavelength, end_wavelength + increment, increment)
    resampled_df = pd.DataFrame({'Wavelength': new_wavelengths})
    new_intensity = np.interp(new_wavelengths, df['Wavelength'], df['Intensity'])

    # Normalize the 'Intensity' column
    resampled_df['Intensity'] = (new_intensity - new_intensity.min()) / (new_intensity.max() - new_intensity.min())

    return resampled_df

def update_plots( spectrum_df, iteration_count ):
    # Plot absorbance vs wavelength and save
    fig = plt.figure(figsize = ( 4 , 3) , dpi = 300 ) 
    gs = fig.add_gridspec( 1 , 1 )
    axs = fig.add_subplot( gs[ 0 , 0 ] )
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.set_xlabel('Wavelength')
    axs.set_ylabel('Absorbance')

    plt.tight_layout()
    axs.plot(spectrum_df['Wavelength'], spectrum_df['Intensity'], color='black', label='Measured Absorbance', lw=1)
    fig.savefig(DATA_UV_DIR_PATH + 'Absorbance_spectrum_iteration_' + str(iteration_count-1) + '.png' )


# Measure absorbance based on sample spectrum and baseline data
def measure_abs(sample_spectrum):
    divisor = sample_spectrum - baseline_data['dark']
    divisor[divisor <= 0] = np.nan  # Replace non-positive values to avoid log10 issues
    absorbance = np.log10((baseline_data['baseline'] - baseline_data['dark']) / divisor)
    absorbance = absorbance.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace NaNs and Infs

    absorbance_df = pd.DataFrame({
        'Wavelength': baseline_data['wavelength'],  # Ensure Wavelength is float
        'Absorbance': absorbance
    })

    filename = f"absorbance_{iteration_count}.txt"
    filepath = os.path.join(DATA_UV_DIR_PATH, filename)
    absorbance_df.to_csv(filepath, index=False)
    print(f"Absorbance data saved to {filepath}")
    
    return absorbance_df

# Function to write a specific set of values into suggested_parameters_2.txt for PREST_2 pump
def write_suggested_parameters_2(x5,x6,x7,x8):
    param_sugg_2 = pd.DataFrame({
        'iteration': [iteration_count + 1],
        'x1': [x5],
        'x2': [x6],
        'x3': [x7],
        'x4': [x8]
    })
    write_header = not os.path.exists(parameters2pump_2_path)
    param_sugg_2.to_csv(parameters2pump_2_path, mode='a', header=write_header, index=False)
    print("Written to suggested_parameters_2.txt")

# Function to write a specific set of values into suggested_parameters.txt for PREST pump
def write_suggested_parameters(x1,x2,x3,x4):
    param_sugg = pd.DataFrame({
        'iteration': [iteration_count + 1],
        'x1': [x1],
        'x2': [x2],
        'x3': [x3],
        'x4': [x4]
    })
    write_header = not os.path.exists(parameters2pump_1_path)
    param_sugg.to_csv(parameters2pump_1_path, mode='a', header=write_header, index=False)
    print("Written to suggested_parameters.txt")

# Define the generation strategy: first aleatory (SOBOL) then FULLYBAYESIAN
gs = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=num_trials_SOBOL, min_trials_observed=1, max_parallelism=1, model_kwargs={"seed": seed_SOBOL}),
    GenerationStep(model=Models.FULLYBAYESIAN, num_trials=-1, max_parallelism=1, model_kwargs={}),
])


# Initialize the Ax client with the generation strategy
ax_client = AxClient(generation_strategy=gs)

# Create the experiment
ax_client.create_experiment(parameters=[
    {"name": "x1", "type": "range", "bounds": [50, 500]}, # AA 50-500
    {"name": "x2", "type": "range", "bounds": [100, 400]}, # Ag 100-400
    {"name": "x3", "type": "range", "bounds": [10, 50]}, # Seeds 10-50
    {"name": "x4", "type": "range", "bounds": [100, 1000]} # HCl 100-1000
], objectives={
        obj1_name: ObjectiveProperties(minimize=True), # MINIMIZE peak position difference
        obj2_name: ObjectiveProperties(minimize=False), # MAXIMAZE intesity ratio of peak by absorbance at 400 nm  
    },
    tracking_metric_names=["max_peak_intensity", "intensity_ratio"])


# Main optimization loop
spectra_files = []

known_files = set(os.listdir(DATA_UV_DIR_PATH))  # List of known files in the folder

folder_path = DATA_UV_DIR_PATH
# Check for new files
# Keep track of processed files
processed_files = set(os.listdir(folder_path))  # Files that are already processed

iteration_count = 0
NetTVol = 0 #Total amont of liquid volume transfere through the cleaning channel
while iteration_count < num_iterations:
    parameterization, trial_index = ax_client.get_next_trial()
    x1, x2, x3, x4 = parameterization.get("x1"), parameterization.get("x2"), parameterization.get("x3"), parameterization.get("x4")

    print(f"Iteration {iteration_count + 1}: Suggested parameters - x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}")

    parameter_history.append([x1, x2, x3, x4])
    iteration_numbers.append(iteration_count + 1)

    param_sugg = pd.DataFrame({
        'iteration': [iteration_count + 1],
        'x1': [x1],
        'x2': [x2],
        'x3': [x3],
        'x4': [x4]
    })

    write_suggested_parameters(10000,0,0,0) # Adds 10 mL of CTAB
    sleep_with_progress(140, "Adds 10 mL of CTAB")  # Complexation time NOTE: EVEN 100 SECONDS WAIT IS ENOUGH #140

    write_suggested_parameters_2(0, 100, 0, 0) # Add 100uL of Au(III)
    sleep_with_progress(300, "Add 100uL of Au(III)")  # Complexation time #300

    write_suggested_parameters_2(x4,0,0,0) # Adds desired amount of HCl x4
    sleep_with_progress(60, f"Adds desired amount ({x4}uL) of HCl") #15

    write_suggested_parameters(0,0,x2,0) # Adds desired amount of Ag(I) x2
    sleep_with_progress(60, f"Adds desired amount ({x2}uL) of Ag(I)") #15

    write_suggested_parameters(0,x1,0,0) # Add desired amount of AA x1
    sleep_with_progress(60, f"Adds desired amount ({x1}uL) of A.A.") #15

    write_suggested_parameters(0,0,0,x3) # Adds desired amount of Seeds x3
    sleep_with_progress(15, f"Adds desired amount ({x3}uL) of Seeds") #15

    sleep_with_progress(3600, "Waiting for 1 hours for the reaction to complete") # 5400
    
    write_suggested_parameters_2(0,0,7000,0) # This will remove 7000 uL of reaction solution to the cuvette for recording the spectra
    # TAKING THE SPECTRUM OF THE REACTION MIXTURE 
    sleep_with_progress(150, "Waiting for solution to reach the cuvette") #150

    print('Taking the spectrum of the reaction mixture...')
    sample_spectrum = spec.intensities()
    measure_abs(sample_spectrum)
    #sample_spectrum = process_spectrum(sample_spectrum)

    # CHECK THE ABOVE CODE BLOCK AND MAKE SURE THAT THE SAMPLE_SPECTRUM IS DEFINED AND WILL BE USED IN THE NEXT BLOCK.

    while True:
        files = os.listdir(folder_path)  # List of all files in the directory
            # Find new .txt files that haven't been processed
        new_files = [f for f in files if f.endswith(".txt") and f not in processed_files]
            
        if new_files:
                # Assign the first new file detected to spectrum_filename
            spectrum_filename = new_files[0]  # This assigns the new file added to the folder
            print(f"New file detected: {spectrum_filename}")

            try:
                # Attempt to read the new spectrum file into a DataFrame
                spectrum_filepath = os.path.join(folder_path, spectrum_filename)
                new_spectrum_df_1 = pd.read_csv(spectrum_filepath, delimiter=',')
                new_spectrum_df = process_spectrum(new_spectrum_df_1)
                    # Process the new spectrum file (if required, use your existing process_spectrum function here)
                print(new_spectrum_df_1.head())  # Print the first few rows to verify it's loaded correctly
                print(new_spectrum_df.head())  # Print the first few rows to verify it's loaded correctly
                processed_files.add(spectrum_filename)  # Add this file to the processed list
                break  # Exit the loop after processing the new file

            except Exception as e:
                print(f"Error reading file {spectrum_filename}: {e}")

        time.sleep(1)  # Check every second for new files

        # Now you have the processed DataFrame 'new_spectrum_df'

    # Calculate the result
    result = objective_function(new_spectrum_df)
    peak_position_values.append(result[obj1_name])
    intensity_ratio_values.append(result[obj2_name])

    ax_client.complete_trial(trial_index=trial_index, raw_data={
        obj1_name: result["peak_position_diff"], 
        obj2_name: result["intensity_ratio"]
    })
    print('Experiment trial complete...')

    update_plots(new_spectrum_df, iteration_count + 1)

    print('Begin pump cleaning...')

    write_suggested_parameters_2(0,0,7000,0)
    sleep_with_progress(150, "Waiting for removing solution from the reactor") #150
    write_suggested_parameters_2(0,0,0,10000)
    sleep_with_progress(150, "Waiting to fill clean water in the reactor") #150
    print('Cleaning second time...')
    write_suggested_parameters_2(0,0,15000,0)
    sleep_with_progress(200, "Waiting for removing solution from the reactor") #200
    write_suggested_parameters_2(0,0,0,10000)
    sleep_with_progress(150, "Waiting to fill clean water in the reactor") #150
    print('Cleaning third time...')
    write_suggested_parameters_2(0,0,15000,0)
    sleep_with_progress(200, "Waiting for removing solution from the reactor") #200
    write_suggested_parameters_2(0,0,0,10000)
    sleep_with_progress(150, "Waiting to fill clean water in the reactor") #150
    print('Making the reactor ready for next iteration...')
    write_suggested_parameters_2(0,0,15000,0)
    sleep_with_progress(200, "Waiting for removing solution from the reactor") #200

    NetTVol = NetTVol + 40200 + x1 + x2 + x3 + x4
    print( 'Total amont of liquid volume transfere through the cleaning channel:' + str( NetTVol / 1000 ) + ' ml' )
    
    iteration_count += 1

#plt.ioff()  # Turn off interactive mode
#plt.show()

print("All iterations completed.")

df_res = ax_client.get_trials_data_frame()
print('Results',  df_res)
df_res.to_csv(DATA_UV_DIR_PATH + 'parameters_objective_result.csv', index=False)








