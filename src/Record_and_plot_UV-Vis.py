import seabreeze.spectrometers as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from colorama import Fore

# Load baseline and dark spectra
df = pd.read_csv("spectrometer_data.txt", sep='\t')
wavelengths = df['wavelength'].values
dark = df['dark'].values
baseline = df['baseline'].values

# Connect to spectrometer
devices = sb.list_devices()
if devices:
    spec = sb.Spectrometer(devices[0])
    print("Spectrophotometer connected.")
else:
    print("No spectrophotometer found!")
    exit()

spec.integration_time_micros(40000)  # 40 ms

# Record current spectrum
input("Press Enter to record spectrum with the sample in place...")
intensities = spec.intensities()

# Subtract dark, normalize
corrected = intensities - dark
corrected = corrected - baseline
normalized = corrected / (baseline - dark + 1e-9)

# Create DataFrame
df_out = pd.DataFrame({
    "Wavelength (nm)": wavelengths,
    "Raw": intensities,
    "Corrected": corrected,
    "Normalized": normalized
})

# Save CSV
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_csv = f"uv_vis_spectrum_{timestamp}.txt"
df_out.to_csv(output_csv, index=False, sep='\t')
print(Fore.GREEN + f"Spectrum data saved to '{output_csv}'")

# Plot and save PNG
plt.figure(figsize=(8, 4))
plt.plot(wavelengths, baseline, color='black', linewidth=0.8)
plt.title("UV-Vis Spectrum (Normalized)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity")
plt.xlim(300, 1100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_img = f"uv_vis_spectrum_{timestamp}.png"
plt.savefig(output_img, dpi=300)
#plt.show()
print(Fore.CYAN + f"Plot saved to '{output_img}'")
