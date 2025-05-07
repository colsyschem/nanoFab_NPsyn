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
print(Fore.GREEN + 'Successfully imported all the packages.')

# Connect to the spectrophotometer
devices = sb.list_devices()
if not devices:
    print(Fore.RED + 'No Spectrophotometer Found!')
    exit()
spec = sb.Spectrometer(devices[0])
print(Fore.GREEN + 'Spectrophotometer successfully connected.')
spec.integration_time_micros(40000)

# Initialize lists to store data for plotting
loss_values = []
iteration_numbers = []
parameter_history = []

obj1_name = "syn_param"

def plot_loss(loss, iteration_count):
    fig.ion()
    fig, ax = plt.subplots()

    ax.plot(iteration_count, loss)
    ax.xlabel('Iterations')
    ax.ylabel('Loss')
    ax.spines([['top', 'right']]).set_visible(False)

    plt.draw()
    plt.pause(0.001)
    fig.ioff()

def plot_suggested_parameters(suggested_param, iteration_count):
    suggested_param = param_df

    fig.ion()
    fig, ax = plt.subplots(4,1, sharex= True)

    ax[0].plot(iteration_count, suggested_param[0])
    ax[0].ylabel('Param1')
    ax[1].plot(iteration_count, suggested_param[1])
    ax[1].ylabel('Param2')
    ax[2].plot(iteration_count, suggested_param[2])
    ax[2].ylabel('Param3')
    ax[3].plot(iteration_count, suggested_param[3])
    ax[3].ylabel('Param4')
    ax[3].xlabel('Iterations')


# I want to plot ref_df and spectrum value in a dynamic fashion as in it should keep updating while in the loop
# I want at the end of the expeimrnt to make a subplot and put all the values there.
# I want to make a loss vs iteration graph 
# I want to make a sample parameters vs iteration graph
# I want all these graphs to be saved in the results folder with a proper date etc. 


# Initialize the plots outside the loop
def initialize_plots():
    fig, axs = plt.subplots(6, 1, figsize=(10, 18))

    axs[0].set_title('Spectrum Comparison')
    axs[0].set_xlabel('Data Point Index')
    axs[0].set_ylabel('Intensity')

    axs[1].set_title('Loss vs Iterations')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')

    axs[2].set_title('Suggested Parameter x1 vs Iteration')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('x1')

    axs[3].set_title('Suggested Parameter x2 vs Iteration')
    axs[3].set_xlabel('Iteration')
    axs[3].set_ylabel('x2')

    axs[4].set_title('Suggested Parameter x3 vs Iteration')
    axs[4].set_xlabel('Iteration')
    axs[4].set_ylabel('x3')

    axs[5].set_title('Suggested Parameter x4 vs Iteration')
    axs[5].set_xlabel('Iteration')
    axs[5].set_ylabel('x4')

    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    return fig, axs

# Function to update plots during the loop
def update_plots(fig, axs, ref_df, generated_spectrum, iteration_count, loss_values, parameter_history):
    params_array = np.array(parameter_history)

    # Clear the previous plot data
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[3].cla()
    axs[4].cla()
    axs[5].cla()

    # Plot reference and generated spectrum
    axs[0].plot(ref_df.index, ref_df.iloc[:, 0], label='Reference Spectrum', color='blue')
    axs[0].plot(generated_spectrum.index, generated_spectrum.iloc[:, 0], label=f'Generated Spectrum (Iter {iteration_count})', color='orange')
    axs[0].legend()

    # Plot loss vs iterations
    axs[1].plot(iteration_numbers, loss_values, marker='o', color='green')

    # Plot suggested parameters vs iterations
    axs[2].plot(iteration_numbers, params_array[:, 0], marker='o', label='x1', color='blue')
    axs[3].plot(iteration_numbers, params_array[:, 1], marker='o', label='x2', color='orange')
    axs[4].plot(iteration_numbers, params_array[:, 2], marker='o', label='x3', color='green')
    axs[5].plot(iteration_numbers, params_array[:, 3], marker='o', label='x4', color='red')

    # Redraw and pause
    fig.canvas.draw()
    plt.pause(0.1)













# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR', '')
folder_path = DATA_UV_DIR_PATH
# Get the path to the folder containing reference spectrum
REF_SPECTRUM_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'reference_spectrum', '')
# ------------------------- output folder -----------------------
# Get the path to the folder for saving suggested parameters
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src', '')
OUTPUT_DIR_PATH_2 = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src', '')

# Load baseline data
baseline_data = pd.read_csv('spectrometer_data.csv')
baseline_data['wavelength'] = baseline_data['wavelength'].astype(float)  # Ensure baseline wavelength is float
iteration_count = 0 
num_iterations = 10
# Measure absorbance based on sample spectrum and baseline data
def measure_abs(sample_spectrum, iteration_count=0):
    # Handle division to avoid invalid log values
    divisor = sample_spectrum - baseline_data['dark']
    divisor[divisor <= 0] = np.nan  # Replace non-positive values to avoid log10 issues
    absorbance = np.log10((baseline_data['baseline'] - baseline_data['dark']) / divisor)
    absorbance = absorbance.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace NaNs and Infs

    # Create DataFrame with Wavelength and Absorbance columns
    absorbance_df = pd.DataFrame({
        'Wavelength': baseline_data['wavelength'],  # Ensure Wavelength is float
        'Absorbance': absorbance
    })
    
    # Save absorbance data
    filename = f"absorbance_{iteration_count}.csv"
    filepath = os.path.join(DATA_UV_DIR_PATH, filename)
    absorbance_df.to_csv(filepath, index=False)
    print(f"Absorbance data saved to {filepath}")
    
    return absorbance_df  # Return the DataFrame with Wavelength and Absorbance

# Function to clean and resample the reference data
def clean_and_resample_absorbance(df, start_wavelength=300, end_wavelength=1100, interval=1):
    """
    Clean the DataFrame by removing NaN and -inf values, and resample between
    specified wavelengths with a given interval.

    Parameters:
    df (pd.DataFrame): DataFrame with 'Wavelength' and 'Absorbance' columns.
    """
    # Remove NaN and -inf values from the DataFrame
    df = df.replace([np.nan, -np.inf, np.inf], np.nan).dropna()
    
    # Filter for the range between specified start and end wavelengths
    df = df[(df['Wavelength'] >= start_wavelength) & (df['Wavelength'] <= end_wavelength)]
    
    # Generate new wavelength range as float64
    new_wavelengths = pd.DataFrame({
        'Wavelength': np.arange(start_wavelength, end_wavelength + 1, interval, dtype=np.float64)
    })

    # Merge and interpolate
    resampled_data = pd.merge_asof(new_wavelengths, df.astype({'Wavelength': np.float64}), on='Wavelength', direction='nearest')
    resampled_data['Absorbance'] = resampled_data['Absorbance'].interpolate(method='linear')

    return resampled_data

# Measure and process absorbance
#input('Press Y to measure sample spectrum.')
#sample_spectrum = spec.intensities()
#absorbance_df = measure_abs(sample_spectrum)

# Clean and resample, then plot
#absorb = clean_and_resample_absorbance(absorbance_df)
#plt.plot(absorb['Wavelength'], absorb['Absorbance'])
#plt.xlabel("Wavelength (nm)")
#plt.ylabel("Absorbance")
#plt.title("Resampled Absorbance Spectrum")
#plt.show(block=False)
#plt.pause(0.001)

# Define the loss function
def loss_func(ref_df, filename):
    # Read the CSV file
    calc_df = pd.read_csv(filename, header=None, delimiter=',')
    # Perform calculations
    y = round(np.linalg.norm(calc_df.iloc[:, 0] - ref_df.iloc[:, 0]), 2)
    return y

# Set to keep track of processed files for the entire process
processed_files = set()

# Define the generation strategy
gs = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=1, min_trials_observed=1, max_parallelism=1, model_kwargs={"seed": 123}),
    GenerationStep(model=Models.FULLYBAYESIAN, num_trials=-1, max_parallelism=3, model_kwargs={"num_samples": 256, "warmup_steps": 512}),
])

# Initialize the Ax client with the generation strategy
ax_client = AxClient(generation_strategy=gs)

# Create the experiment
ax_client.create_experiment(parameters=[
    {"name": "x1", "type": "range", "bounds": [100, 1000]},
    {"name": "x2", "type": "range", "bounds": [100, 1000]},
    {"name": "x3", "type": "range", "bounds": [100, 1000]},
    {"name": "x4", "type": "range", "bounds": [100, 1000]},
], objectives={obj1_name: ObjectiveProperties(minimize=True)}, parameter_constraints=["x1 + x2 + x3 + x4 <= 2000" , "x1 + x2 + x3 + x4 >= 1900"])

# Define the text file path for saving suggested parameters
parameters_txt_path = os.path.join(OUTPUT_DIR_PATH, 'suggested_parameters.txt')

# Function to generate a random spectrum file based on optimizer parameters
def generate_spectrum_file(file_path, params):
    spectrum_data = np.random.random(100) * params.get("x1") / 100  # Mock spectrum generation logic
    pd.DataFrame(spectrum_data).to_csv(file_path, header=False, index=False)

# Main optimization loop
def run_optimizer():
    global iteration_count
    spectra_files = []

    # Load reference spectrum
    ref_df = pd.read_csv(os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt'), header=None, delimiter=',')

    # Initialize plots
    fig, axs = initialize_plots()

    # Loop until the required number of iterations is reached
    while iteration_count < num_iterations:
        # Get the next trial
        parameterization, trial_index = ax_client.get_next_trial()
        # Extract parameter values
        x1, x2, x3, x4 = parameterization.get("x1"), parameterization.get("x2"), parameterization.get("x3"), parameterization.get("x4")
        
        # Log suggested parameters
        print(f"Iteration {iteration_count + 1}: Suggested parameters - x1: {x1}, x2: {x2}, x3: {x3}, x4: {x4}")

        # Save parameters to history
        parameter_history.append([x1, x2, x3, x4])
        iteration_numbers.append(iteration_count + 1)
        
        # Create a DataFrame for the current iteration's suggested parameters
        param_sugg = pd.DataFrame({
            'iteration': [iteration_count + 1],
            'x1': [x1],
            'x2': [x2],
            'x3': [x3],
            'x4': [x4]
        })
        
        # Append the DataFrame to the text file
        write_header = not os.path.exists(parameters_txt_path)
        param_sugg.to_csv(parameters_txt_path, mode='a', header=write_header, index=False)

        # Generate a new spectrum file based on the optimizer parameters
        spectrum_filename = f"spectrum_{iteration_count + 1}.txt"
        spectrum_filepath = os.path.join(DATA_UV_DIR_PATH, spectrum_filename)
        generate_spectrum_file(spectrum_filepath, parameterization)
        spectra_files.append(spectrum_filepath)

        # Process the new txt file
        result = loss_func(ref_df, spectrum_filepath)  # Calculate spectral difference
        print(f"New file created: {spectrum_filename}. Calculating spectral difference...")

        # Save loss value
        loss_values.append(result)

        # Complete the trial with the calculated loss
        ax_client.complete_trial(trial_index=trial_index, raw_data={obj1_name: result})
        print("Experiment trial completed.")

        # Remember the processed file
        processed_files.add(spectrum_filename)

        # Update plots dynamically
        update_plots(fig, axs, ref_df, generated_spectrum, iteration_count + 1, loss_values, parameter_history)

        # Move to the next iteration
        iteration_count += 1
    
    #plot_spectra(ref_df, spectra_files)

    plt.ioff()  # Turn off interactive mode
    plt.show()

    print("All iterations completed.")

# Function to plot the spectra
def plot_spectra(ref_df, spectra_files):

    plt.ion()
    fig, axs = plt.subplots(10, 5, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle('Output Spectra vs. Reference Spectrum')

    for i, filepath in enumerate(spectra_files):
        ax.clear()
        ax = axs[i // 5, i % 5]
        generated_spectrum = pd.read_csv(filepath, header=None, delimiter=',')
        ax.plot(ref_df, label='Reference Spectrum', color='blue')
        ax.plot(generated_spectrum, label=f'Iteration {i + 1}', color='orange')
        ax.legend(loc='upper right')
        plt.draw()
        plt.pause(1)
    plt.ioff()
    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

# Test function
def test_optimizer():
    # Ensure necessary directories exist
    os.makedirs(DATA_UV_DIR_PATH, exist_ok=True)
    os.makedirs(REF_SPECTRUM_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    # Create a mock reference spectrum
    ref_spectrum_data = np.random.random(100)
    pd.DataFrame(ref_spectrum_data).to_csv(os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt'), header=False, index=False)

    # Run the optimization script
    run_optimizer()

    # Check if the output file is created
    assert os.path.exists(parameters_txt_path), "Output file was not created."

    # Check if the output file contains data
    output_df = pd.read_csv(parameters_txt_path)
    assert not output_df.empty, "Output file is empty."

    print("Test completed successfully.")

# Uncomment to run the test function
run_optimizer()


best_parameters, metrics = ax_client.get_best_parameters()
print(Fore.GREEN + 'The best parameter are: ' + best_parameters + metrics)

print('The script was executed successfully')

