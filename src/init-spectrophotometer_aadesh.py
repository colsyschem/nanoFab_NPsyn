import seabreeze.spectrometers as sb
import pandas as pd
import numpy as np
import os
from colorama import Fore

# --- Connect to the spectrophotometer ---
devices = sb.list_devices()
if devices:
    spec = sb.Spectrometer(devices[0])
    print('Spectrophotometer successfully connected.')
else:
    print('No Spectrophotometer Found!')
    exit()

spec.integration_time_micros(20000)  # 40 ms integration time

# --- Function to average multiple scans ---
def average_scans(spec, num_scans=50):
    print(f"Taking {num_scans} scans and averaging...")
    scans = [spec.intensities() for _ in range(num_scans)]
    return np.mean(scans, axis=0)

# --- Record Baseline ---
record_baseline = input('Press Y to record the baseline: ').strip().lower()
if record_baseline == 'y':
    baseline = average_scans(spec, num_scans=50)
    print("Baseline recorded.")
else:
    print("Baseline recording skipped.")
    exit()

# --- Record Dark Spectrum ---
record_dark = input('Turn off the lamp and press Y to record the dark spectrum: ').strip().lower()
if record_dark == 'y':
    dark = average_scans(spec, num_scans=50)
    print("Dark spectrum recorded. (Don't forget to turn ON the lamp again!)")
else:
    print("Dark spectrum recording skipped.")
    exit()

# --- Save Wavelengths + Data ---
wavelengths = spec.wavelengths()
data = {
    "wavelength": wavelengths,
    "dark": dark,
    "baseline": baseline
}
df = pd.DataFrame(data)

# --- Save to file ---
output_file_path = os.path.join("spectrometer_data_50num_scans.txt")
df.to_csv(output_file_path, index=False, sep='\t')  # Tab-delimited
print(f" Data successfully saved to '{output_file_path}'.")

