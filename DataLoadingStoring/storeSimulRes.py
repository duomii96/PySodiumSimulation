import h5py
import numpy as np
from pathlib import Path
#from Classes.getFunction import getFunction

def save_complex(grp, name, arr):
    """Store a complex array as two real datasets: name (real) and name_im (imag)."""
    arr = np.asarray(arr)
    grp.create_dataset(name, data=np.real(arr).astype(np.float64))
    grp.create_dataset(name + "_im", data=np.imag(arr).astype(np.float64))

def save_TQTPPI(filename, PC, acqTimeVec, mqSpectrum, mqFID, FIDs, tevos, popt=None):
    folder_path = Path(r"C:\Users\dz8\MatlabStuff\Scripte\PaperSimulation\SimulationDataH5\TQTPPI")

    # 2. Combine the folder path with the filename you passed in
    full_path = folder_path / filename

    with h5py.File(full_path, "a") as f:  # 'a' so multiple sequences can be appended to same file
        if "TQTPPI" in f:
            del f["TQTPPI"]  # remove old group so re-runs don't collide
        seq = f.create_group("TQTPPI")
        # --- Simulated data ---
        data = seq.require_group("data")
        save_complex(data, "mqFID", mqFID)            # (2048,) complex
        save_complex(data, "mqSpectrum", mqSpectrum)   # (2048,) complex
        save_complex(data, "FIDs", FIDs)               # (2048, NumPhaseCycles) complex, phase-cycled FIDs

        # --- Time / axis arrays ---
        time = seq.require_group("time")
        time.create_dataset("acqTimeVec", data=acqTimeVec)   # FID acquisition time axis
        time.create_dataset("tevos", data=tevos)              # evolution time axis (indirect dim)
        #time.create_dataset("fAngles90", data=fAngles90)      # phase-cycle flip angles used

        # --- Fit results (optional, if available) ---
        if popt is not None:
            seq.create_dataset("fit_popt", data=np.asarray(popt))

        # --- Metadata / NMR pulse parameters ---
        meta = seq.require_group("meta")
        meta.attrs["sequence"]        = "TQTPPIWith180"
        meta.attrs["dwelltimeFID"]    = PC.dwelltimeFID
        meta.attrs["dataPoints"]      = PC.dataPoints
        meta.attrs["NumPhaseCycles"]  = PC.NumPhaseCycles
        meta.attrs["nSpins"]          = PC.nSpins
        meta.attrs["tmix"]            = PC.tmix
        meta.attrs["flip90"]          = PC.flip90
        meta.attrs["TR"]              = PC.TR
        meta.attrs["tevo"]            = PC.tevo
        meta.attrs["tevo0"]           = PC.tevo0
        meta.attrs["tevoStep"]        = PC.tevoStep





def save_IRTQTPPI(filename, PC, mqFID, mqSpectrum, tevos):
    folder_path = Path(r"C:\Users\dz8\MatlabStuff\Scripte\PaperSimulation\SimulationDataH5\IRTQTPPI")

    # 2. Combine the folder path with the filename you passed in
    full_path = folder_path / filename


    with h5py.File(full_path, "a") as f:  # append mode, same file as other sequences
        seq = f.require_group("IRTQTPPI")

        # --- Simulated data ---
        data = seq.require_group("data")
        save_complex(data, "mqFID", mqFID)
        save_complex(data, "mqSpectrum", mqSpectrum)


        # --- Time axis ---
        time_grp = seq.require_group("time")
        time_grp.create_dataset("tevos", data=tevos)

        # --- Metadata / NMR pulse parameters ---
        meta = seq.require_group("meta")
        meta.attrs["sequence"]        = "IRTQTPPI_sr"
        meta.attrs["dwelltimeFID"]    = PC.dwelltimeFID
        meta.attrs["dataPoints"]      = PC.dataPoints
        meta.attrs["NumPhaseCycles"]  = PC.NumPhaseCycles
        meta.attrs["nSpins"]          = PC.nSpins
        meta.attrs["tmix"]            = PC.tmix
        meta.attrs["flip90"]          = PC.flip90
        meta.attrs["TR"]              = PC.TR
        meta.attrs["tevo"]            = PC.tevo
        meta.attrs["tevo0"]           = PC.tevo0
        meta.attrs["tevoStep"]        = PC.tevoStep



