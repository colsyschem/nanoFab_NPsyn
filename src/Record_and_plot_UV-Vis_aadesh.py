import seabreeze.spectrometers as sb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from colorama import Fore

# --- Connect to Spectrometer ---
devices = sb.list_devices()
if devices:
    spec = sb.Spectrometer(devices[0])
    print('Spectrophotometer successfully connected.')
else:
    print('No Spectrophotometer Found!')
    exit()

spec.integration_time_micros(20000)  # Increased integration time for better signal

# --- Load Baseline and Dark from previous file ---
base_file = "spectrometer_data_50num_scans.txt"
if not os.path.exists(base_file):
    print("Baseline/Dark data not found. Run baseline script first.")
    exit()

df_base = pd.read_csv(base_file, sep='\t')
wavelengths = df_base['wavelength'].values
baseline = df_base['baseline'].values
dark = df_base['dark'].values

# --- Average multiple scans to reduce noise ---
num_scans = 50
print(f"Taking {num_scans} scans and averaging...")
scans = [spec.intensities() for _ in range(num_scans)]
averaged_scan = np.mean(scans, axis=0)

# Subtract dark from both sample and baseline
sample_corrected = averaged_scan - dark
baseline_corrected = baseline - dark

# Avoid division by zero
baseline_corrected[baseline_corrected == 0] = 1e-9

# Calculate absorbance
normalized = sample_corrected / baseline_corrected
normalized = np.clip(normalized, 1e-9, None)  # Avoid log(0)
absorbance = -np.log10(normalized)
absorbance = np.clip(absorbance, 1e-9, None)  # Avoid division by 0 when inverting


# --- Filter the data between 300 and 1100 nm ---
wavelength_mask = (wavelengths >= 300) & (wavelengths <= 1100)
wavelengths_filtered = wavelengths[wavelength_mask]
absorbance_masked = absorbance[wavelength_mask]
absorbance_filtered = absorbance[wavelength_mask]
absorbance_filtered = (absorbance_filtered - absorbance_filtered.min()) / (absorbance_filtered.max() - absorbance_filtered.min())

# print(f"Length of wavelengths_filtered: {len(wavelengths_filtered)}")
# print(f"Length of absorbance_filtered: {len(absorbance_filtered)}")

# --- Save the result ---
output_df = pd.DataFrame({
    'Wavelength (nm)': wavelengths_filtered,
    'Absorbance': absorbance_masked,
    'Absorbance (normalized)': absorbance_filtered
})
txt_filename = "uvvis_corrected_spectrum_50num_scans.txt"
output_df.to_csv(txt_filename, sep='\t', index=False)
print(f"Corrected spectrum saved to '{txt_filename}'.")

# --- Plot the spectrum using ax ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(wavelengths_filtered, absorbance_filtered, color='black', linewidth=0.8)

# Customize plot
#ax.set_title("Corrected UV-Vis Spectrum (300–1100 nm)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Absorbance")
#ax.set_ylim([0, 1])
ax.grid(True, linestyle='--', linewidth=0.5)

# Layout and save
fig.tight_layout()
png_filename = "uvvis_50num_scans.png"
fig.savefig(png_filename, dpi=300)
print( f"Plot saved as '{png_filename}'")
