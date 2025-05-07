import seabreeze.spectrometers as sb
import pandas as pd
import numpy as np
import os
from colorama import Fore

# Connection with the spectrophotometer
devices = sb.list_devices()
if devices:
    spec = sb.Spectrometer(devices[0])
    print('Spectrophotometer successfully connected.')
else:
    print('No Spectrophotometer Found!')
    exit()

spec.integration_time_micros(40000)  # Set integration time of 40ms

# Take the baseline
input_recording = input('Press Y to record the baseline: ').strip().lower()
if input_recording == 'y':
    baseline = spec.intensities()
    print("Baseline recorded.")
else:
    print("Baseline recording skipped.")
    exit()

# Take the dark spectrum with lamp off
input_recording = input('Turn off the lamp and press Y to record the dark spectrum: ').strip().lower()
if input_recording == 'y':
    dark = spec.intensities()
    print("Dark spectrum recorded. Do not forget to turn on the lamp.")
else:
    print("Dark spectrum recording skipped.")
    exit()

# Capture the wavelengths
wavelengths = spec.wavelengths()

# Create a DataFrame
data = {
    "wavelength": wavelengths,
    "dark": dark,
    "baseline": baseline
}
df = pd.DataFrame(data)

# Export the DataFrame to a CSV file
output_file_path = os.path.join("spectrometer_data.txt")
df.to_csv(output_file_path, index=False, sep='\t')  # Save as tab-delimited for .txt
print( f"Data successfully saved to '{output_file_path}'.")
