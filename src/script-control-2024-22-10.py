import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.factory import Models
import seabreeze.spectrometers as sb
from pathlib import Path
from colorama import Fore
import time
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import init_notebook_plotting, render

print(Fore.GREEN + 'Successfully imported all the packages.')

# Connect to the spectrophotometer
devices = sb.list_devices()
if not devices:
    print(Fore.RED + 'No Spectrophotometer Found!')
    # Uncomment if you want to exit during debugging
    # exit()
spec = sb.Spectrometer(devices[0])
print(Fore.GREEN + 'Spectrophotometer successfully connected.')
spec.integration_time_micros(40000)

# Initialize lists to store data for plotting
loss_values = []
iteration_numbers = []
parameter_history = []

obj1_name = "syn_param"

# Initialize the plots outside the loop
def initialize_plots():
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    # Configure each plot's title, labels, and hide top/right spines
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0, 0].set_title('Spectrum Comparison')
    axs[0, 0].set_xlabel('Wavelength (nm)')
    axs[0, 0].set_ylabel('Extinction')

    axs[0, 1].set_title('Loss vs Iterations')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')

    axs[0, 2].set_title('Suggested Parameter x1 vs Iteration')
    axs[0, 2].set_xlabel('Iteration')
    axs[0, 2].set_ylabel('x1')

    axs[1, 0].set_title('Suggested Parameter x2 vs Iteration')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('x2')

    axs[1, 1].set_title('Suggested Parameter x3 vs Iteration')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('x3')

    axs[1, 2].set_title('Suggested Parameter x4 vs Iteration')
    axs[1, 2].set_xlabel('Iteration')
    axs[1, 2].set_ylabel('x4')

    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    return fig, axs

def process_spectrum(df, start_wavelength=300, end_wavelength=1100, increment=1):
    df = df.iloc[14:].reset_index(drop=True)
    df[0] = pd.to_numeric(df[0], errors='coerce')  # First column assumed to be wavelengths
    df[1] = pd.to_numeric(df[1], errors='coerce')  # Second column assumed to be intensities
    df = df.dropna()
    df[1] = (df[1] - df[1].min()) / (df[1].max() - df[1].min())
    new_wavelengths = np.arange(start_wavelength, end_wavelength + increment, increment)
    resampled_df = pd.DataFrame({'Wavelength': new_wavelengths})
    resampled_df['Normalized_Intensity'] = np.interp(new_wavelengths, df[0], df[1])
    return resampled_df

def update_plots(fig, axs, ref_df, new_spectrum_df, iteration_count, loss_values, parameter_history, result):
    params_array = np.array(parameter_history)
    for ax in axs.flat:
        ax.cla()

    axs[0, 0].fill_between(ref_df['Wavelength'], ref_df['Normalized_Intensity'], label='Reference Spectrum', color='royalblue', alpha = 0.5)
    axs[0, 0].fill_between(new_spectrum_df['Wavelength'], new_spectrum_df['Normalized_Intensity'], label=f'New Spectrum (Iter {iteration_count}, Loss: {result})', color='firebrick', alpha = 0.5)
    axs[0, 0].set_xlabel('Wavelength (nm)')
    axs[0, 0].set_ylabel('Extinction')
    axs[0, 0].legend()

    axs[0, 1].plot(iteration_numbers, loss_values, marker='o', color='green')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')

    axs[0, 2].plot(iteration_numbers, params_array[:, 0], marker='o', label='x1', color='blue')
    axs[0, 2].set_xlabel('Iteration')
    axs[0, 2].set_ylabel('x1')
    axs[1, 0].plot(iteration_numbers, params_array[:, 1], marker='o', label='x2', color='orange')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('x2')
    axs[1, 1].plot(iteration_numbers, params_array[:, 2], marker='o', label='x3', color='green')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('x3')
    axs[1, 2].plot(iteration_numbers, params_array[:, 3], marker='o', label='x4', color='red')
    axs[1, 2].set_xlabel('Iteration')
    axs[1, 2].set_ylabel('x4')

    fig.canvas.draw()
    plt.pause(0.5)

# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
folder_path = DATA_UV_DIR_PATH
REF_SPECTRUM_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'reference_spectrum', '')
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')

# Load baseline data
baseline_data = pd.read_csv('spectrometer_data.txt', delimiter = '\t')
baseline_data['wavelength'] = baseline_data['wavelength'].astype(float)

iteration_count = 0 
num_iterations = 10

# Measure absorbance based on sample spectrum and baseline data
def measure_abs(sample_spectrum, iteration_count=0):
    divisor = sample_spectrum - baseline_data['dark']
    divisor[divisor <= 0] = np.nan  # Replace non-positive values to avoid log10 issues
    absorbance = np.log10((baseline_data['baseline'] - baseline_data['dark']) / divisor)
    absorbance = absorbance.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace NaNs and Infs

    absorbance_df = pd.DataFrame({
        'Wavelength': baseline_data['wavelength'],
        'Absorbance': absorbance
    })
    
    filename = f"absorbance_{iteration_count}.txt"
    filepath = os.path.join(DATA_UV_DIR_PATH, filename)
    absorbance_df.to_csv(filepath, index=False)
    print(f"Absorbance data saved to {filepath}")
    
    return absorbance_df

# Loss function
def loss_func(ref_df, new_spectrum_df):
    y = round(np.linalg.norm(new_spectrum_df['Normalized_Intensity'] - ref_df['Normalized_Intensity']), 2)
    return y

# Define the generation strategy
gs = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=4, min_trials_observed=3, max_parallelism=1, model_kwargs={"seed": 123}),
    GenerationStep(model=Models.FULLYBAYESIAN, num_trials=-1, max_parallelism=1, model_kwargs={"num_samples": 256, "warmup_steps": 512}),
])

# Initialize the Ax client with the generation strategy
ax_client = AxClient(generation_strategy=gs)

# Create the experiment
ax_client.create_experiment(parameters=[
    {"name": "x1", "type": "range", "bounds": [100, 1000]},
    {"name": "x2", "type": "range", "bounds": [100, 1000]},
    {"name": "x3", "type": "range", "bounds": [100, 1000]},
    {"name": "x4", "type": "range", "bounds": [100, 1000]},
], objectives={obj1_name: ObjectiveProperties(minimize=True)}, parameter_constraints=["x1 + x2 + x3 + x4 <= 2000", "x1 + x2 + x3 + x4 >= 1995"])

# Define the text file path for saving suggested parameters
parameters_txt_path = os.path.join(OUTPUT_DIR_PATH, 'suggested_parameters.txt')

# Main optimization loop
spectra_files = []

ref_df = pd.read_csv(os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt'), skiprows=14, header=None, delimiter='\t')
ref_df = process_spectrum(ref_df)
print(Fore.RED + 'REF. SPECTRUM', ref_df)

fig, axs = initialize_plots()

known_files = set(os.listdir(DATA_UV_DIR_PATH))

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

    write_header = not os.path.exists(parameters_txt_path)
    param_sugg.to_csv(parameters_txt_path, mode='a', header=write_header, index=False)

    processed_files = set(os.listdir(folder_path))

    while True:
        files = os.listdir(folder_path)
        new_files = [f for f in files if f.endswith(".txt") and f not in processed_files]

        if new_files:
            spectrum_filename = new_files[0]
            print(f"New file detected: {spectrum_filename}")

            try:
                spectrum_filepath = os.path.join(folder_path, spectrum_filename)
                new_spectrum_df = pd.read_csv(spectrum_filepath, header=None, skiprows=14, delimiter='\t')
                new_spectrum_df = process_spectrum(new_spectrum_df)
                print(new_spectrum_df.head())
                processed_files.add(spectrum_filename)
                break

            except Exception as e:
                print(f"Error reading file {spectrum_filename}: {e}")

        time.sleep(1)

    result = loss_func(ref_df, new_spectrum_df)
    print(f"Calculating spectral difference for spectrum file: {new_spectrum_df}")

    loss_values.append(result)

    ax_client.complete_trial(trial_index=trial_index, raw_data=result)
    print("Experiment trial completed.")

    update_plots(fig, axs, ref_df, new_spectrum_df, iteration_count + 1, loss_values, parameter_history, result)

    iteration_count += 1

plt.ioff()
plt.show()
print("All iterations completed.")

best_parameters, metrics = ax_client.get_best_parameters()
print(Fore.GREEN + f'The best parameters are: {best_parameters}')
print(Fore.GREEN + f'The best metrics are: {metrics}')
