import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from pathlib import Path
from colorama import Fore
print(Fore.GREEN + 'Successfully imported all the packages.')

# Define paths
BASE_DIR_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_UV_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'UV-Vis-NIR')
REF_SPECTRUM_DIR_PATH = os.path.join(BASE_DIR_PATH, 'data', 'reference_spectrum')
OUTPUT_DIR_PATH = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST', 'src')
OUTPUT_DIR_PATH_2 = os.path.join(BASE_DIR_PATH, 'hardware', 'PREST_2', 'src')

# Initialize iteration count and number of iterations
iteration_count = 0
num_iterations = 10
obj1_name = "syn_param"
objective_name = 'syn_param'

# Define the loss function
def loss_func(ref_df, calc_df):
    y = round(np.linalg.norm(calc_df.iloc[:, 0] - ref_df.iloc[:, 0]), 2)
    return y

# Initialize lists to store data for plotting
loss_values = []
iteration_numbers = []
parameter_history = []

# Set to keep track of processed files
processed_files = set()

# Define the generation strategy
gs = GenerationStrategy(steps=[
    GenerationStep(model=Models.SOBOL, num_trials=1, min_trials_observed=1, max_parallelism=1, model_kwargs={"seed": 123}),
    GenerationStep(model=Models.GPEI, num_trials=-1, max_parallelism=3),
])

# Initialize the Ax client with the generation strategy
ax_client = AxClient(generation_strategy=gs)

# Create the experiment
ax_client.create_experiment(
    parameters=[
        {"name": "x1", "type": "range", "bounds": [100, 1000]},
        {"name": "x2", "type": "range", "bounds": [100, 1000]},
        {"name": "x3", "type": "range", "bounds": [100, 1000]},
        {"name": "x4", "type": "range", "bounds": [100, 1000]},
    ],
    objectives={"syn_param": ObjectiveProperties(minimize=True)},
    parameter_constraints=["x1 + x2 + x3 + x4 <= 2000", "x1 + x2 + x3 + x4 >= 1900"]
)

# Define the text file path for saving suggested parameters
parameters_txt_path = os.path.join(OUTPUT_DIR_PATH, 'suggested_parameters.txt')

# Function to generate a random spectrum file based on optimizer parameters
def generate_spectrum_file(file_path, params):
    spectrum_data = np.random.random(100) * params.get("x1") / 100  # Mock spectrum generation logic
    pd.DataFrame(spectrum_data).to_csv(file_path, header=False, index=False)

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

# Main optimization loop
def run_optimizer():
    global iteration_count
    spectra_files = []

    # Load reference spectrum
    ref_df = pd.read_csv(os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt'), header=None)

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
        generated_spectrum = pd.read_csv(spectrum_filepath, header=None)
        result = loss_func(ref_df, generated_spectrum)  # Calculate spectral difference
        print(f"Iteration {iteration_count + 1}: Loss = {result}")

        # Save loss value
        loss_values.append(result)

        # Complete the trial with the calculated loss
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        print("Experiment trial completed.")

        # Update plots dynamically
        update_plots(fig, axs, ref_df, generated_spectrum, iteration_count + 1, loss_values, parameter_history)

        # Move to the next iteration
        iteration_count += 1

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Test function
def test_optimizer():
    # Ensure necessary directories exist
    os.makedirs(DATA_UV_DIR_PATH, exist_ok=True)
    os.makedirs(REF_SPECTRUM_DIR_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

    # Create a mock reference spectrum if it doesn't exist
    ref_spectrum_path = os.path.join(REF_SPECTRUM_DIR_PATH, 'ref_spectrum.txt')
    if not os.path.exists(ref_spectrum_path):
        ref_spectrum_data = np.random.random(100)
        pd.DataFrame(ref_spectrum_data).to_csv(ref_spectrum_path, header=False, index=False)

    # Run the optimization script
    run_optimizer()

    # Check if the output file is created
    assert os.path.exists(parameters_txt_path), "Output file was not created."

    # Check if the output file contains data
    output_df = pd.read_csv(parameters_txt_path)
    assert not output_df.empty, "Output file is empty."

    print("Test completed successfully.")

# Run the test function
test_optimizer()

# Get best parameters
best_parameters, values = ax_client.get_best_parameters()
print(Fore.GREEN + f"The best parameters are: {best_parameters}")
print(Fore.GREEN + f"Best objective value: {values}")

print('The script was executed successfully')