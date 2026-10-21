# inspect_dictionary.py
# Overview of dictionary contents: input T2 values, fitted T2* values, and gamma
# (Cauchy scale of wShiftdist [rad/s]; older dictionaries stored FWHM = 2*gamma).

import h5py
import numpy as np
import matplotlib.pyplot as plt

from dict_params import gamma_column

#fname = 'pyDict_5090FWHM.h5'
#fname = 'dictionary_adjFWHM30100.h5'
fname = 'dict_general_gamma2045.h5'
#fname = 'proteins_dict_gamma3045.h5'
# ── Load ──────────────────────────────────────────────────────────────────
try:
    with h5py.File(fname, 'r') as f:
        params  = f['/params'][:]    # [nEntries x 3]: [T2f, T2s, gamma (old: FWHM)]
        T2f_fit = f['/T2ffit'][:]   # [nEntries x 1]  T2f* in seconds
        T2s_fit = f['/T2sfit'][:]   # [nEntries x 1]  T2s* in seconds
        attrs   = dict(f.attrs)

except:
    with h5py.File(fname, 'r') as f:
        params  = f['/params'][:]
        params = params.transpose()# [nEntries x 3]: [T2f, T2s, FWHM]
        T2f_fit = f['/T2f_fit'][:]   # [nEntries x 1]  T2f* in seconds
        T2s_fit = f['/T2s_fit'][:]   # [nEntries x 1]  T2s* in seconds
        attrs   = dict(f.attrs)

# h5py reads in C-order (row-major); transpose if needed to match MATLAB column layout
# Assuming params shape is (nEntries, 3)
T2f_in = params[:, 0] * 1e3  # input T2f  (ms)
T2s_in = params[:, 1] * 1e3  # input T2s  (ms)
gamma  = gamma_column(params, attrs)  # gamma (rad/s)
T2f_s  = T2f_fit.ravel() * 1e3  # fitted T2f* (ms)
T2s_s  = T2s_fit.ravel() * 1e3  # fitted T2s* (ms)

nEntries   = params.shape[0]
gamma_vals = np.unique(gamma)
ngamma     = len(gamma_vals)

print(f"Dictionary: {nEntries} entries  |  {ngamma} gamma values: "
      f"{' '.join(f'{v:.1f}' for v in gamma_vals)} rad/s")
print(f"T2f  input:  {T2f_in.min():.2f} - {T2f_in.max():.2f} ms")
print(f"T2s  input:  {T2s_in.min():.1f} - {T2s_in.max():.1f} ms")
print(f"T2f* fitted: {T2f_s.min():.2f} - {T2f_s.max():.2f} ms")
print(f"T2s* fitted: {T2s_s.min():.1f} - {T2s_s.max():.1f} ms")

# Color cycle mimicking MATLAB's lines()
colors = plt.cm.tab10(np.linspace(0, 1, max(ngamma, 1)))

# ── Figure 1: Input vs Fitted T2* scatter per gamma ──────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.8), num='Input T2 vs Fitted T2*')

ax = axes1[0]
for iF, gamma_val in enumerate(gamma_vals):
    mask = gamma == gamma_val
    ax.scatter(T2f_in[mask], T2f_s[mask], s=30, color=colors[iF],
               label=f'gamma={gamma_val:.1f}rad/s')
ax.plot([0, T2f_in.max()], [0, T2f_in.max()], 'k--', label='identity')
ax.set_xlabel('T2f input (ms)'); ax.set_ylabel('T2f* fitted (ms)')
ax.set_title('Fast component')
ax.legend(loc='upper left'); ax.grid(True); ax.axis('tight')

ax = axes1[1]
for iF, gamma_val in enumerate(gamma_vals):
    mask = gamma == gamma_val
    ax.scatter(T2s_in[mask], T2s_s[mask], s=30, color=colors[iF],
               label=f'gamma={gamma_val:.1f}rad/s')
ax.plot([0, T2s_in.max()], [0, T2s_in.max()], 'k--', label='identity')
ax.set_xlabel('T2s input (ms)'); ax.set_ylabel('T2s* fitted (ms)')
ax.set_title('Slow component')
ax.legend(loc='upper left'); ax.grid(True); ax.axis('tight')

fig1.suptitle('Dictionary: input T2 vs fitted T2*')
fig1.tight_layout()

# ── Figure 2: T2* reduction due to gamma (broadening effect) ─────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2), num='T2* reduction vs gamma')

ax = axes2[0]
for iF, gamma_val in enumerate(gamma_vals):
    mask = gamma == gamma_val
    ax.scatter(T2f_in[mask], T2f_in[mask] - T2f_s[mask], s=30, color=colors[iF],
               label=f'gamma={gamma_val:.1f}rad/s')
ax.set_xlabel('T2f input (ms)'); ax.set_ylabel('T2f - T2f* (ms)')
ax.set_title('Fast component broadening')
ax.legend(); ax.grid(True)

ax = axes2[1]
for iF, gamma_val in enumerate(gamma_vals):
    mask = gamma == gamma_val
    ax.scatter(T2s_in[mask], T2s_in[mask] - T2s_s[mask], s=30, color=colors[iF],
               label=f'gamma={gamma_val:.1f}rad/s')
ax.set_xlabel('T2s input (ms)'); ax.set_ylabel('T2s - T2s* (ms)')
ax.set_title('Slow component broadening')
ax.legend(); ax.grid(True)

fig2.suptitle('T2* reduction relative to T2 input (should increase with gamma)')
fig2.tight_layout()

# ── Figure 3: 2D scatter of T2f* and T2s* for each gamma ─────────────────
fig3, axes3 = plt.subplots(1, ngamma, figsize=(3 * ngamma, 4.2), num='T2* grid per gamma')
if ngamma == 1:
    axes3 = [axes3]

for iF, gamma_val in enumerate(gamma_vals):
    mask = gamma == gamma_val
    ax = axes3[iF]
    ax.scatter(T2f_s[mask], T2s_s[mask], s=30, color=colors[iF])
    ax.set_xlabel('T2f* (ms)'); ax.set_ylabel('T2s* (ms)')
    ax.set_title(f'gamma = {gamma_val:.1f} rad/s\n({mask.sum()} entries)')
    ax.grid(True); ax.axis('tight')

fig3.suptitle('Fitted T2f* vs T2s* coverage per gamma value')
fig3.tight_layout()

# ── Table: summary statistics per gamma ──────────────────────────────────
header = f"{'gamma':<10}  {'n':<8}  {'T2f min':<8}  {'T2f max':<8}  " \
         f"{'T2f*min':<8}  {'T2f*max':<8}  {'T2s min':<8}  {'T2s max':<8}"
print(f"\n{header}")
print('-' * 78)
for gamma_val in gamma_vals:
    mask = gamma == gamma_val
    print(f"{gamma_val:<10.1f}  {mask.sum():<8d}  "
          f"{T2f_in[mask].min():<8.2f}  {T2f_in[mask].max():<8.2f}  "
          f"{T2f_s[mask].min():<8.2f}  {T2f_s[mask].max():<8.2f}  "
          f"{T2s_in[mask].min():<8.1f}  {T2s_in[mask].max():<8.1f}")
print("\nAll times in ms, gamma in rad/s.")

plt.show()
