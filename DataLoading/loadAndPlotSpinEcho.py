from pathlib import Path

import numpy as np

from DataLoading.NEOreadIn import NEOreadInSE
from commonImports import *
from Classes.getFunction import getFunction
from pathlib import Path
from DataLoading.NEOreadIn import NEOreadInSE
from commonImports import *

# 1. Base path pointing to the parent folder of the repetition numbers
basePath = Path(r"J:\AG_XNMR\14_Reichert\Messdaten\RawData\NEO\ZI\SR_SR_SEvTQTPPI_BrukerLin_1_9_20230623_165629")

# Define the range of folder numbers (e.g., 28, 29, 30, 31)
# (28,32)
# 182,185
folder_numbers = range(312,315)

# List to temporarily store the magnitude curves for averaging
all_reps_magnitudes = []

# Loop over each repetition folder
for num in folder_numbers:
    dataPath = basePath / str(num)
    print(f"Processing folder: {dataPath}")

    # Load data for this repetition
    complexData, method, acqp = NEOreadInSE(dataPath)

    numEchoes = method['NumInvTime']
    echoTimes = method['InvTime']

    # Remove leading ringing artefact from FID
    complexFIDs_cut = complexData[:, 77:]


    timeVector = np.arange(complexFIDs_cut.shape[-1]) * 0.1

    # Create SE FID, by taking the max value around the possible peak area
    expected_peaks = (echoTimes // 2)  # Starting point, in points

    window_half_width = 10 # in pts
    window_size = 2 * window_half_width + 1
    offsets = np.arange(-window_half_width, window_half_width + 1)[None, :]

    # Calculate window indices (cast to integer immediately)
    window_indices = (expected_peaks[:, None] + offsets).astype(int)

    # Prevent out-of-bounds indexing by clipping to array boundaries
    window_indices = np.clip(window_indices, 0, complexFIDs_cut.shape[-1] - 1)

    # Use dynamically fetched numEchoes instead of hardcoded 256
    row_indices = np.arange(numEchoes)[:, None].astype(int)

    # Extract windowed data
    windowed_data = complexFIDs_cut[row_indices, window_indices]  # Shape: (numEchoes, window_size)

    # Use Real component as requested
    real_window = np.abs(windowed_data)
    real_magnitudes = np.max(real_window, axis=1)

    # Append this repetition's result to our temporary list
    all_reps_magnitudes.append(real_magnitudes)

# 2. Average the data across all repetitions
# Stack lists into a 2D array of shape (number_of_folders, numEchoes)
stacked_magnitudes = np.vstack(all_reps_magnitudes)

# Take the mean along axis 0 (averaging over the repetitions)
averaged_magnitudes = np.mean(stacked_magnitudes, axis=0)

# --- Verification & Plotting ---
print(f"\nStacked shape (Repetitions, Echoes): {stacked_magnitudes.shape}")  # Should be (4, 256)
print(f"Averaged shape: {averaged_magnitudes.shape}")  # Should be (256,)

plt.figure(figsize=(8, 5))
plt.plot(echoTimes, averaged_magnitudes, label="Averaged Signal", color='blue', linewidth=2)
plt.title("Spin Echo ")
plt.xlabel("Echo Time")
plt.ylabel(" Amplitude [a.u.]")
plt.legend()
plt.grid(True)
plt.show()


# ------------- Fit Spin Echo----------------------

echoTimes *= 1e-03
p0 = [35e-03, 8e-03,0.5,0]
p0 = [35e-03, 3e-03,0.5,0.5,0]
bounds = (
    [0, 0, 0.4, -0.1],
    [0.07, 0.070, 0.6, 0.1]
)
popt, pcov = getFunction.fit_biexponentialSQ(echoTimes, averaged_magnitudes, p0 = p0, plot=True, singleAmp=False)
print(popt)