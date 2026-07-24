from commonImports import fftshift, fft, ifftshift, ifft, plt, np, real, imag
from NEOimportTQTPPIspectro import importTQTPPIspec
#from fixedEvoTiEval import fixedEvoTimesEval
#from sigFit import FitTQTPPI, fixedFitTQTPPI
#from fixedEvoTiEval import fixedEvoTimesEval, fitBoth
#from dataToFroFile import toFile
from query_dictionary import query_dictionary
from pathlib import Path
import os


def getSQTQ_fixAcqsDim(path_to_study):
    # Signal processing flags
    pre_filter = 0;    filter_fac_pre = 1
    filter_2nd_dim = 0;     filter_fac_2nd_dim = 1
    filter_fid = 0;    filter_fac_post = 1
    pre_dc_comp = 1
    spike_comp = 1
    post_dc_comp = 1
    w0corr = 1
    zero_fill = 1
    only_real = 1
    is_tau_mixed = 0
    phase_corr = 1

    freqDriftVal = [0, 0, 0]

    data_path = f"{path_to_study}{os.sep}"

    (method, raw_data, complex_data_all_phase, complex_data_unw, real_data_all_phase,
     complex_data_all_phase_unw, mq_fid, mq_spectra, mix_time, evo_time) = importTQTPPIspec(data_path, spike_comp,
                                                                                            phase_corr, pre_dc_comp,
                                                                                            filter_2nd_dim,
                                                                                            filter_fac_2nd_dim,
                                                                                            pre_filter, filter_fac_pre,
                                                                                            w0corr, freqDriftVal,
                                                                                            post_dc_comp, filter_fid,
                                                                                            filter_fac_post, only_real)

    NR = method['Repetitions']


    idx_TQ = method['NumPhaseCycles']  # 0-based: NumPhaseCycles+1 in MATLAB -> index NumPhaseCycles
    idx_SQ = method['NumPhaseCycles'] * 3  # 0-based: NumPhaseCycles*3+1 -> index NumPhaseCycles*3

    if NR == 1:
        complex_data1 = np.squeeze(complex_data_all_phase)


        ft_acq = np.fft.fftshift(np.fft.fft(np.real(complex_data1), axis=0), axes=0)
        tq_peaks = np.real(ft_acq[idx_TQ, :])
        sq_peaks = np.real(ft_acq[idx_SQ, :])
        # adjust for Bruker offset
        sq_peaks = np.append(sq_peaks[5:], sq_peaks[-5:])


    else:

            complex_data1 = np.mean(complex_data_all_phase, axis=0)
            mq_fid_tmp = np.mean(mq_fid, axis=1)
            mq_spectra_tmp = np.mean(mq_spectra, axis=1)


            ft_acq = np.fft.fftshift(np.fft.fft(np.real(complex_data1), axis=0), axes=0)
            tq_peaks = np.real(ft_acq[idx_TQ, :])
            sq_peaks = np.real(ft_acq[idx_SQ, :])
            # adjust for Bruker offset
            sq_peaks = np.append(sq_peaks[5:], sq_peaks[-5:])

    # Expand data to 2048 data points by mirroring the last data points
    # get current size:
    diff_points = 2048 - len(sq_peaks)
    sq_peaks = np.append(sq_peaks, sq_peaks[-diff_points:])
    tq_peaks = np.append(tq_peaks, tq_peaks[-diff_points:])

    sq_peaks = sq_peaks / np.amax(sq_peaks)
    tq_peaks = tq_peaks / np.amax(tq_peaks)

    evo_time = evo_time * 1e-3 # convert ms -> s


    return sq_peaks, tq_peaks, evo_time