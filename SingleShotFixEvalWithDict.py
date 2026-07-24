import matplotlib.pyplot as plt

from commonImports import fftshift, fft, ifftshift, ifft, plt, np, real, imag
from NEOimportTQTPPIspectro import importTQTPPIspec
#from fixedEvoTiEval import fixedEvoTimesEval
#from sigFit import FitTQTPPI, fixedFitTQTPPI
#from fixedEvoTiEval import fixedEvoTimesEval, fitBoth
#from dataToFroFile import toFile
import pywt
from query_dictionary import query_dictionary
from pathlib import Path
import os


def wavelet_denoising(data, wavelet="db4", level=1):
    # 1. Decompose the signal into wavelet coefficients
    # coefficients format: [cA_n, cD_n, cD_n-1, ..., cD_1]
    # cA are approximations (low frequency), cD are details (high frequency noise)
    coeff = pywt.wavedec(data, wavelet, mode="per")

    # 2. Calculate a universal threshold based on the noise variance
    # We look at the finest detail coefficients (cD_1) to estimate noise level
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # 3. Apply soft thresholding to all detail coefficients
    # We skip the very first element (coeff[0]) because it contains the core signal structure
    denoised_coeff = [coeff[0]] + [
        pywt.threshold(i, value=threshold, mode="soft") for i in coeff[1:]
    ]

    # 4. Reconstruct the cleaned signal
    clean_signal = pywt.waverec(denoised_coeff, wavelet, mode="per")
    return clean_signal





IsFixed = 0  # Default


#studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_ThimoAgar_nReco_1_44_20260119_160948")
#studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_metHem4_rapidVol_thimo_1_65_20260514_130257")
studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_hem6_rapidVol_thimo_1_63_20260512_082106")
#studyPath = Path("G:/Transfer/Dominik/DataTemp/DZ_Agar2Temperature_1_86_20240708_084113")

# studyPath = basePath + '/DZ_MedSup_ismrm24_1_37_20231104_115236/'
# startFolder, stopFolder, IsFixed = 94, 172, 1
# studyPath = basePath + '/DZ_DZ_CS_Test_BrukerLin_1_6_20230110_111523/'  # BSA
# studyPath = basePath + '/DZ_SpCoilTest_1_52_20240209_110111/'

start_folder, stop_folder, is_fixed = 51, 55, 1

dict_evo_times = [1, 2, 5, 10,11,12,13,14, 15, 20, 25, 30]
#dict_evo_times = [ 30]

#dict_name = "pyDict_50100FWHM_new.h5"
dict_name = "proteins_dict_6090FWHM.h5"
#dict_name = 'dictionary_adjFWHM30100.h5'

T2s_gt = 18.5
T2s_gt_err = 1.8
T2f_gt = 5.3
T2f_gt_err = 1.2

denoise_TQ = 0

FWHM, FWHM_tol = 80,10
norm_to_match = "2"

#skip_indices    = list(range(33, 60))   # [12:1:19] inclusive
skip_indices    = []
# Signal processing flags
pre_filter        = 0;  filter_fac_pre      = 1
filter_2nd_dim    = 0;  filter_fac_2nd_dim  = 1
filter_fid        = 0;  filter_fac_post     = 1
pre_dc_comp       = 1
spike_comp        = 1
post_dc_comp      = 1
w0corr            = 1
zero_fill         = 1
only_real         = 1
is_tau_mixed      = 0
phase_corr        = 1

# ── Pre-process skip list ─────────────────────────────────────────────────
skip_indices = sorted(set(
    k for k in skip_indices if start_folder < k < stop_folder
))

# ── Result accumulators ───────────────────────────────────────────────────
tau          = {}   # j -> evoTime float
evo_times    = {}   # j -> method.EvoTime
mq_fid1      = {}
mq_spectra1  = {}
tq_peaks_temp = {}
sq_peaks_temp = {}

max_T2s        = []
max_T2f_to_slow = []
mean5_T2f      = []
mean5_T2s      = []
std5_T2s_list  = []
std5_T2f_list  = []
best_T2s       = []
best_T2f       = []
evo_times_list = []

j = 1   # repetition index
i = 0   # result storage index


for k in np.arange(start_folder, stop_folder + 1):
    if k in skip_indices:
        continue
    path_temp = studyPath / f"{k}"
    data_path = f"{path_temp}{os.sep}"

    freqDriftVal = [0, 0, 0]

    (method, raw_data, complex_data_all_phase, complex_data_unw, real_data_all_phase,
     complex_data_all_phase_unw, mq_fid, mq_spectra, mix_time, evo_time) = importTQTPPIspec(data_path, spike_comp, phase_corr, pre_dc_comp, filter_2nd_dim, filter_fac_2nd_dim,
                     pre_filter, filter_fac_pre, w0corr, freqDriftVal, post_dc_comp, filter_fid, filter_fac_post, only_real)

    NR = method['Repetitions']

    if evo_time not in dict_evo_times:
        continue

    idx_TQ = method['NumPhaseCycles']  # 0-based: NumPhaseCycles+1 in MATLAB -> index NumPhaseCycles
    idx_SQ = method['NumPhaseCycles'] * 3  # 0-based: NumPhaseCycles*3+1 -> index NumPhaseCycles*3

    if NR == 1:
        complex_data1 = np.squeeze(complex_data_all_phase)
        mq_fid1[j] = mq_fid
        mq_spectra1[j] = mq_spectra
        evo_times[j] = evo_time


        ft_acq = np.fft.fftshift(np.fft.fft(np.real(complex_data1), axis=0), axes=0)
        tq_peaks = np.real(ft_acq[idx_TQ, :])
        sq_peaks = np.real(ft_acq[idx_SQ, :])
        # adjust for Bruker offset
        sq_peaks = np.append(sq_peaks[5:],sq_peaks[-5:])
        tq_peaks_temp[j] = tq_peaks
        sq_peaks_temp[j] = sq_peaks
        j += 1

    else:
        if is_fixed:
            complex_data1 = np.mean(complex_data_all_phase, axis=0)
            mq_fid_tmp = np.mean(mq_fid, axis=1)
            mq_spectra_tmp = np.mean(mq_spectra, axis=1)
            mq_fid1[j] = mq_fid_tmp
            mq_spectra1[j] = mq_spectra_tmp
            evo_times[j] = evo_time

            ft_acq = np.fft.fftshift(np.fft.fft(np.real(complex_data1), axis=0), axes=0)
            tq_peaks = np.real(ft_acq[idx_TQ, :])
            sq_peaks = np.real(ft_acq[idx_SQ, :])
            # adjust for Bruker offset
            sq_peaks = np.append(sq_peaks[5:], sq_peaks[-5:])

            tq_peaks_temp[j] = tq_peaks
            sq_peaks_temp[j] = sq_peaks
            j += 1

        else:
            for r in range(NR):
                mq_fid1[j + r] = mq_fid[:, r]
                mq_spectra1[j + r] = mq_spectra[:, r]
            tau[j] = evo_time
            evo_ti = (j - 1) // NR
            evo_times[evo_ti] = method.EvoTime
            j += NR




    spacing = int(method["NumPhaseCycles"]) * 2

    # ── Acquisition time vector ───────────────────────────────────────────
    x_vector = np.arange(1, len(tq_peaks) + 1) * 1e-4  # seconds

    # ── Dictionary matching ───────────────────────────────────────────────
    if denoise_TQ:
        print("Denoising TQ signal using wavelet decomposition! \n")
        tq_denoised = wavelet_denoising(tq_peaks, level = 1)
        #tq_peaks = tq_denoised

        result = query_dictionary(dict_name, sq_peaks, tq_denoised, evo_time, norm_to_match, FWHM, FWHM_tol)
    else:
        result = query_dictionary(dict_name, sq_peaks, tq_peaks, evo_time, norm_to_match,FWHM, FWHM_tol)

    top5_T2s = result.top10_T2s[:5]
    top5_T2f = result.top10_T2f[:5]

    idx_max_T2s = int(np.argmax(top5_T2s))
    max_T2s.append(top5_T2s[idx_max_T2s])
    max_T2f_to_slow.append(top5_T2f[idx_max_T2s])

    mean5_T2f.append(np.mean(top5_T2f))
    mean5_T2s.append(np.mean(top5_T2s))
    best_T2s.append(result.top10_T2s[0])
    best_T2f.append(result.top10_T2f[0])
    evo_times_list.append(evo_time)

    std5_T2s_list.append(float(np.std(top5_T2s, ddof=1)))  # always a plain Python float
    std5_T2f_list.append(float(np.std(top5_T2f, ddof=1)))

    # ── Plot: measured vs best match ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    t_ms = x_vector * 1e3
    ax1.plot(t_ms, sq_peaks / np.abs(sq_peaks).max(), label='measured')
    ax1.plot(result.acqTime * 1e3,
             result.SQ_match / np.abs(result.SQ_match).max(),
             '--', label='best match')
    ax1.set(title=f"SQ; EvoTime: {evo_time} ms", xlabel='ms')
    ax1.legend();
    ax1.grid(True)

    ax2.plot(t_ms, tq_peaks / np.abs(tq_peaks).max(), label='measured')
    ax2.plot(result.acqTime * 1e3,
             result.TQ_match / np.abs(result.TQ_match).max(),
             '--', label='best match')
    ax2.set(title='TQ', xlabel='ms')
    ax2.legend();
    ax2.grid(True)

    fig.tight_layout()
    plt.show()

    print(f"Highest possible T2f: {result.top10_T2f[0].item():4.2f} ms")
    print(f"Lowest possible T2s:  {result.top10_T2s[0].item():4.2f}")

    i += 1
    #tqPeaks, sqPeaks, ratio, noise_var = fixedEvoTimesEval(ftAllshaped, evoTime, spacing, secondDimFit=False)

    """pSQ, pTQ = fitBoth(sqPeaks, tqPeaks)
    # params contain Cov's in second dim, i.e. pSQ[0,:]=pOpt, pSQ[1,:]=pCov
    initIdx = np.argmax(sqPeaks)
    sqAdjust = sqPeaks[initIdx:]

    Tqtimes = "T2f={}, T2s={}".format(np.round(pTQ[0, 1], 3), np.round(pTQ[0, 2], 3))
    # create time vector, with correction for initial adc deadtime and spike.
    timeVec = np.arange(len(sqAdjust)) / 10.
    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
    ax[0].plot(timeVec, sqPeaks[initIdx:] / np.amax(real(sqPeaks)), label='Signal')
    ax[0].plot(timeVec, fixedFitTQTPPI.f_11(timeVec, pSQ[0, 0], pSQ[0, 1], pSQ[0, 2], pSQ[0, 3], pSQ[0, 4]))
    ax[0].set_title('SQ-Signal')
    ax[0].set_xlabel('time [ms]')
    ax[0].axvline(x=evoTime, ymin=0, ymax=1., color='salmon', linewidth=1.5, label=f'EvoTime: {evoTime} ms')
    ax[0].legend(loc="best")
    ax[1].set_title('TQ-Signal')
    ax[1].plot(timeVec, tqPeaks[initIdx:] / np.amax(tqPeaks), label='Signal')
    ax[1].plot(timeVec, fixedFitTQTPPI.f_31(timeVec, pTQ[0, 0], pTQ[0, 1], pTQ[0, 2], pTQ[0, 3]), label=Tqtimes)
    ax[1].axvline(x=evoTime, ymin=np.min(tqPeaks), ymax=np.max(tqPeaks), color='salmon', linewidth=1.5,
                  label=f'EvoTime: {evoTime} ms')
    ax[1].legend(loc="best")
    plt.show()
"""
evo_times_arr   = np.array(evo_times_list)
max_T2s_arr     = np.array(max_T2s)
max_T2f_arr     = np.array(max_T2f_to_slow)
best_T2s_arr    = np.array(best_T2s)
best_T2f_arr    = np.array(best_T2f)


def plot_dict_vs_gt(
    evo_times: np.ndarray,
    T2s_vals: np.ndarray,
    T2f_vals: np.ndarray,
    std_T2s: float,
    std_T2f: float,
    T2s_gt: float,
    T2f_gt: float,
    T2s_gt_err: float,
    T2f_gt_err: float,
    title: str,
) -> plt.Figure:
    """
    Plot dictionary T2s/T2f results against ground truth with shaded error bounds.

    Parameters
    ----------
    evo_times  : evolution time points (ms)
    T2s_vals   : T2s estimates at each evo time
    T2f_vals   : T2f estimates at each evo time
    std_T2s    : error bar size for T2s
    std_T2f    : error bar size for T2f
    T2s_gt     : ground truth T2s (ms)
    T2f_gt     : ground truth T2f (ms)
    T2s_gt_err : uncertainty on T2s ground truth
    T2f_gt_err : uncertainty on T2f ground truth
    title      : figure title string
    """
    t_range = [evo_times.min() - 0.5, evo_times.max() + 0.5]

    fig, ax = plt.subplots(figsize=(8, 5))

    # 1. Shaded error bounds
    ax.fill_between(
        t_range,
        [T2s_gt - T2s_gt_err] * 2,
        [T2s_gt + T2s_gt_err] * 2,
        color=(0.9, 0.9, 0.9), alpha=0.5, edgecolor='none'
    )
    ax.fill_between(
        t_range,
        [T2f_gt - T2f_gt_err] * 2,
        [T2f_gt + T2f_gt_err] * 2,
        color=(0.9, 0.8, 0.8), alpha=0.5, edgecolor='none'
    )

    # 2. Ground truth lines
    ax.hlines(T2s_gt, *t_range, colors='k', linewidths=1.5, linestyles='--')
    ax.hlines(T2f_gt, *t_range, colors='k', linewidths=1.5, linestyles='--')

    # 3. Measured data with error bars
    ax.errorbar(evo_times.ravel(), T2s_vals.ravel(), yerr=float(np.array(std_T2s).ravel()[0]), fmt='o-', linewidth=1,
                label='T2s dict.')
    ax.errorbar(evo_times.ravel(), T2f_vals.ravel(), yerr=float(np.array(std_T2f).ravel()[0]), fmt='o-', linewidth=1,
                label='T2f dict.')

    # Dummy handles for GT legend entries (shaded patches excluded from auto-legend)
    ax.plot([], [], 'k--', label='T2s GT')
    ax.plot([], [], 'k--', label='T2f GT')

    ax.set_xlabel('Evolution time [ms]')
    ax.set_ylabel(r'$T_{2s/f}$')
    ax.set_title(title)
    ax.legend(
        handles=[
            ax.get_lines()[2],   # T2s GT dashed line
            ax.get_lines()[3],   # T2f GT dashed line
            ax.containers[0],    # T2s errorbar
            ax.containers[1],    # T2f errorbar
        ],
        labels=['T2s GT', 'T2f GT', 'T2s dict.', 'T2f dict.'],
        loc='best'
    )
    ax.grid(True)
    fig.tight_layout()
    return fig



# ── Call for both figures ─────────────────────────────────────────────────
fig1 = plot_dict_vs_gt(
    evo_times_arr, max_T2s_arr, max_T2f_arr,
    np.array(std5_T2s_list).ravel(), np.array(std5_T2f_list).ravel(),
    T2s_gt, T2f_gt, T2s_gt_err, T2f_gt_err,
    title='MaxDict(T2s/f)'
)

fig2 = plot_dict_vs_gt(
    evo_times_arr, best_T2s_arr, best_T2f_arr,
    std5_T2s_list, std5_T2f_list,
    T2s_gt, T2f_gt, T2s_gt_err, T2f_gt_err,
    title='BestDict(T2s/f)'
)

plt.show()
# Recompute final std over top-5 across all iterations (last iteration values)
# If you need per-iteration std stored, accumulate them in a list similarly.

"""# ── Summary Figure 1: max T2s / paired T2f ───────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(evo_times_arr, max_T2s_arr,   fmt='o-', label='max T2s (top-5)')
ax.errorbar(evo_times_arr, max_T2f_arr,   fmt='o-', label='T2f paired to max T2s')
ax.set(xlabel='Evolution time (ms)', ylabel='T2 (ms)')
ax.legend(); ax.grid(True)
fig1.tight_layout()

# ── Summary Figure 2: best-match T2s / T2f ───────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(evo_times_arr, best_T2s_arr, fmt='o-', label='best T2s')
ax.errorbar(evo_times_arr, best_T2f_arr, fmt='o-', label='best T2f')
ax.set(xlabel='Evolution time (ms)', ylabel='T2 (ms)')
ax.legend(); ax.grid(True)
fig2.tight_layout()

plt.show()"""