# inspect_dictionary.py
# Overview of dictionary contents: input T2 values, fitted T2* values, and FWHM.

import h5py
import numpy as np
import matplotlib.pyplot as plt

#fname = 'pyDict_5090FWHM.h5'
#fname = 'dictionary_adjFWHM30100.h5'
fname = 'proteins_dict_6090FWHM.h5'
# ── Load ──────────────────────────────────────────────────────────────────
try:
    with h5py.File(fname, 'r') as f:
        params  = f['/params'][:]    # [nEntries x 3]: [T2f, T2s, FWHM]
        T2f_fit = f['/T2ffit'][:]   # [nEntries x 1]  T2f* in seconds
        T2s_fit = f['/T2sfit'][:]   # [nEntries x 1]  T2s* in seconds

except:
    with h5py.File(fname, 'r') as f:
        params  = f['/params'][:]
        params = params.transpose()# [nEntries x 3]: [T2f, T2s, FWHM]
        T2f_fit = f['/T2f_fit'][:]   # [nEntries x 1]  T2f* in seconds
        T2s_fit = f['/T2s_fit'][:]   # [nEntries x 1]  T2s* in seconds

# h5py reads in C-order (row-major); transpose if needed to match MATLAB column layout
# Assuming params shape is (nEntries, 3)
T2f_in = params[:, 0] * 1e3  # input T2f  (ms)
T2s_in = params[:, 1] * 1e3  # input T2s  (ms)
FWHM = params[:, 2]  # FWHM       (Hz)
T2f_s  = T2f_fit.ravel() * 1e3  # fitted T2f* (ms)
T2s_s  = T2s_fit.ravel() * 1e3  # fitted T2s* (ms)

nEntries  = params.shape[0]
FWHM_vals = np.unique(FWHM)
nFWHM     = len(FWHM_vals)

print(f"Dictionary: {nEntries} entries  |  {nFWHM} FWHM values: "
      f"{' '.join(f'{v:.0f}' for v in FWHM_vals)} Hz")
print(f"T2f  input:  {T2f_in.min():.2f} - {T2f_in.max():.2f} ms")
print(f"T2s  input:  {T2s_in.min():.1f} - {T2s_in.max():.1f} ms")
print(f"T2f* fitted: {T2f_s.min():.2f} - {T2f_s.max():.2f} ms")
print(f"T2s* fitted: {T2s_s.min():.1f} - {T2s_s.max():.1f} ms")

# Color cycle mimicking MATLAB's lines()
colors = plt.cm.tab10(np.linspace(0, 1, max(nFWHM, 1)))

# ── Figure 1: Input vs Fitted T2* scatter per FWHM ───────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.8), num='Input T2 vs Fitted T2*')

ax = axes1[0]
for iF, fwhm_val in enumerate(FWHM_vals):
    mask = FWHM == fwhm_val
    ax.scatter(T2f_in[mask], T2f_s[mask], s=30, color=colors[iF],
               label=f'FWHM={fwhm_val:.0f}Hz')
ax.plot([0, T2f_in.max()], [0, T2f_in.max()], 'k--', label='identity')
ax.set_xlabel('T2f input (ms)'); ax.set_ylabel('T2f* fitted (ms)')
ax.set_title('Fast component')
ax.legend(loc='upper left'); ax.grid(True); ax.axis('tight')

ax = axes1[1]
for iF, fwhm_val in enumerate(FWHM_vals):
    mask = FWHM == fwhm_val
    ax.scatter(T2s_in[mask], T2s_s[mask], s=30, color=colors[iF],
               label=f'FWHM={fwhm_val:.0f}Hz')
ax.plot([0, T2s_in.max()], [0, T2s_in.max()], 'k--', label='identity')
ax.set_xlabel('T2s input (ms)'); ax.set_ylabel('T2s* fitted (ms)')
ax.set_title('Slow component')
ax.legend(loc='upper left'); ax.grid(True); ax.axis('tight')

fig1.suptitle('Dictionary: input T2 vs fitted T2*')
fig1.tight_layout()

# ── Figure 2: T2* reduction due to FWHM (broadening effect) ──────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2), num='T2* reduction vs FWHM')

ax = axes2[0]
for iF, fwhm_val in enumerate(FWHM_vals):
    mask = FWHM == fwhm_val
    ax.scatter(T2f_in[mask], T2f_in[mask] - T2f_s[mask], s=30, color=colors[iF],
               label=f'FWHM={fwhm_val:.0f}Hz')
ax.set_xlabel('T2f input (ms)'); ax.set_ylabel('T2f - T2f* (ms)')
ax.set_title('Fast component broadening')
ax.legend(); ax.grid(True)

ax = axes2[1]
for iF, fwhm_val in enumerate(FWHM_vals):
    mask = FWHM == fwhm_val
    ax.scatter(T2s_in[mask], T2s_in[mask] - T2s_s[mask], s=30, color=colors[iF],
               label=f'FWHM={fwhm_val:.0f}Hz')
ax.set_xlabel('T2s input (ms)'); ax.set_ylabel('T2s - T2s* (ms)')
ax.set_title('Slow component broadening')
ax.legend(); ax.grid(True)

fig2.suptitle('T2* reduction relative to T2 input (should increase with FWHM)')
fig2.tight_layout()

# ── Figure 3: 2D scatter of T2f* and T2s* for each FWHM ──────────────────
fig3, axes3 = plt.subplots(1, nFWHM, figsize=(3 * nFWHM, 4.2), num='T2* grid per FWHM')
if nFWHM == 1:
    axes3 = [axes3]

for iF, fwhm_val in enumerate(FWHM_vals):
    mask = FWHM == fwhm_val
    ax = axes3[iF]
    ax.scatter(T2f_s[mask], T2s_s[mask], s=30, color=colors[iF])
    ax.set_xlabel('T2f* (ms)'); ax.set_ylabel('T2s* (ms)')
    ax.set_title(f'FWHM = {fwhm_val:.0f} Hz\n({mask.sum()} entries)')
    ax.grid(True); ax.axis('tight')

fig3.suptitle('Fitted T2f* vs T2s* coverage per FWHM value')
fig3.tight_layout()

# ── Table: summary statistics per FWHM ───────────────────────────────────
header = f"{'FWHM(Hz)':<10}  {'n':<8}  {'T2f min':<8}  {'T2f max':<8}  " \
         f"{'T2f*min':<8}  {'T2f*max':<8}  {'T2s min':<8}  {'T2s max':<8}"
print(f"\n{header}")
print('-' * 78)
for fwhm_val in FWHM_vals:
    mask = FWHM == fwhm_val
    print(f"{fwhm_val:<10.0f}  {mask.sum():<8d}  "
          f"{T2f_in[mask].min():<8.2f}  {T2f_in[mask].max():<8.2f}  "
          f"{T2f_s[mask].min():<8.2f}  {T2f_s[mask].max():<8.2f}  "
          f"{T2s_in[mask].min():<8.1f}  {T2s_in[mask].max():<8.1f}")
print("\nAll times in ms.")

plt.show()