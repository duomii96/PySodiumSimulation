from commonImports import *
import PyBrukerReadIn as pbr

import scipy


def importTQTPPIspec(data_path, spikeComp, phaseCorr, preDCcomp, filter2ndDim, filterFac2ndDim,
                     preFilter, filterFacPre, w0corr, freqDriftVal, postDCcomp, filterFID, filterFacPost, onlyReal):
    ds = pbr.ReadExperiment(path=data_path)

    rawComplexData = ds.raw_fid
    acqp = ds.acqp
    method = ds.method

    numPhaseCycles, numPhaseSteps, nR = int(method['NumPhaseCycles']), int(method['NumPhaseSteps']), int(
        method['Repetitions'])

    specDim = int(2 ** np.ceil(np.log2(method['PVM_SpecMatrix'])))

    # specDim = 2 ** np.ceil(np.log2(method.get_value('PVM_Matrix')[0]))

    if method['Method'] == '<User:anVsTqtppiSpectro>':
        complexDataAllPhase = np.reshape(rawComplexData, (specDim, numPhaseCycles * numPhaseSteps * 2, nR))
        tmp = complexDataAllPhase[:, 0::2, :] + complexDataAllPhase[:, 1::2, :]
        complexDataAllPhase = tmp
        del tmp
    else:
        complexDataAllPhase = np.reshape(rawComplexData, (nR, numPhaseCycles * numPhaseSteps, specDim))

    if any(np.real(complexDataAllPhase[0,0,100:10]) < 0):
        complexDataAllPhase = complexDataAllPhase * (-1)

    if np.sum(imag(complexDataAllPhase[0,0,:])) > np.sum(real(complexDataAllPhase[0,0,:])):
        # switch real and imaginary part
        print("Hello MF")
        temp_real = real(complexDataAllPhase)
        temp_imag = imag(complexDataAllPhase)
        complexDataAllPhase = temp_imag + temp_real * 1j
    ###### Spike Compensation ##############
    if spikeComp:
        for i in range(nR):
            for idx in range(complexDataAllPhase.shape[1]):
                tmp1 = np.concatenate(
                    (complexDataAllPhase[i, idx, :75], np.flipud(complexDataAllPhase[i, idx, :76]) * -1))
                tmp2 = complexDataAllPhase[i, idx, :151]
                tmp3 = tmp2.flatten() - tmp1.flatten()
                complexDataAllPhase[i, idx, :151] = tmp3.squeeze()
    complexDataAllPhase = complexDataAllPhase[:, :, 76:]
    complexDataAllPhase[:, :, 0] = complexDataAllPhase[:, :, 0] * 0.5
    # end
    tmp1, tmp2, tmp3 = None, None, None

    # -------- 05 Phase correction ------------------

    complexDataUnw = np.zeros(np.shape(complexDataAllPhase))

    if phaseCorr:
        # for k = 1:method.Repetitions
        #   for h = 1:method.NumPhaseCycles
        # complexDataAllPhaseUnw = complexDataAllPhase * exp(-1i * angle(complexDataAllPhase(1)));
        complexfft = np.fft.fftshift(np.fft.fft(complexDataAllPhase[0, 0, :]))

        pos = np.argmax(abs(complexfft))
        # [~,pos] = max(abs(complexfft)); % alternative dazu ist direkt die Index info aus max fkt zu benutzen --> ist auch schneller
        # apple = @(x) abs(imag(complexfft(pos,1)*exp(1i*x)));
        banana = lambda x: -np.real(complexfft[pos] * np.exp(1j * x))
        x, fval = scipy.optimize.fmin(banana, [0], full_output=True)[:2]

        complexDataAllPhaseUnw = complexDataAllPhase * np.exp(1j * x)

    else:
        complexDataAllPhaseUnw = complexDataAllPhase

    # ---- 06 preDCcomp -------------------

    if preDCcomp:
        complexDataAllPhaseUnw[0, :] *= 0.5

    # ---- 07 Filter 2nd Dimension --------------------

    if filter2ndDim:
        timeVecCos = np.linspace(0, 1 / 2 * np.pi, int(np.fix(complexDataAllPhase.shape[1] / filterFac2ndDim)))
        nPoints = int(np.fix(complexDataAllPhase.shape[1] * 0.1))

        for k in range(nR):
            for idx in range(complexDataAllPhase.shape[0]):
                complexDataAllPhase[k, :, idx] -= np.mean(complexDataAllPhase[k, -nPoints:, idx])
                complexDataAllPhase[k, :len(timeVecCos), idx] *= np.cos(timeVecCos) ** 2
                complexDataAllPhase[k, len(timeVecCos):, idx] = 0
                complexDataAllPhaseUnw[k, :, idx] -= np.mean(complexDataAllPhaseUnw[k, -nPoints:, idx])
                complexDataAllPhaseUnw[k, :len(timeVecCos), idx] *= np.cos(timeVecCos) ** 2
                complexDataAllPhaseUnw[k, len(timeVecCos):, idx] = 0

    # ---- 08 preFilter ----------------------------

    # cos²-filter along the time dimension
    if preFilter:
        timeVecCos = np.linspace(0, 1 / 2 * np.pi, int(np.floor(complexDataAllPhase.shape[-1] / filterFacPre)))
        for k in range(nR):
            for idx in range(complexDataAllPhase.shape[1]):
                complexDataAllPhase[k, idx, :len(timeVecCos)] = complexDataAllPhase[k, idx, :len(timeVecCos)] * (
                        np.cos(timeVecCos) ** 2)
                complexDataAllPhase[k, idx, len(timeVecCos):] = 0
                complexDataAllPhaseUnw[k, idx, :len(timeVecCos)] = complexDataAllPhaseUnw[k, idx, :len(timeVecCos)] * (
                        np.cos(timeVecCos) ** 2)
                complexDataAllPhaseUnw[k, idx, len(timeVecCos):] = 0
    #
    ## correct resonance frequency of wrapped data
    # print('THIRD')
    # noClearVars = whos()
    for idxRep in range(nR):
        for idxPhase in range(1):  # range(size(complexDataAllPhase,2))
            magDataAllPhase = np.abs(
                np.real(np.fft.fftshift(np.fft.fft(np.real(complexDataAllPhase[idxRep, idxPhase, :])))))
            numPoints = len(complexDataAllPhase[idxRep, idxPhase, :])
            maxMagDataAllPhase = np.argmax(magDataAllPhase[:int(numPoints / 2) + 1])
            max0 = maxMagDataAllPhase
            posRamp = 1
            incAllPhase = 0
            corrPhaseAllPhase = 0
            tmpDataAllPhase = complexDataAllPhase[idxRep, idxPhase, :]
            originalPhaseAllPhase = np.angle(tmpDataAllPhase[0])
            if w0corr:
                if maxMagDataAllPhase < (int(numPoints / 2) + 1) == 1:
                    while maxMagDataAllPhase < (int(numPoints / 2) + 1) or incAllPhase == 1:
                        corrTmpDataAllPhase = tmpDataAllPhase * np.exp(
                            posRamp * 1j * incAllPhase * (180 / numPoints) * np.arange(1, numPoints + 1))
                        magCorrTmpDataAllPhase = np.abs(
                            np.real(np.fft.fftshift(np.fft.fft(np.real(corrTmpDataAllPhase)))))
                        newPhaseAllPhase = np.angle(corrTmpDataAllPhase[0])
                        corrPhaseAllPhase = newPhaseAllPhase - originalPhaseAllPhase
                        corrTmpDataAllPhase = corrTmpDataAllPhase * np.exp(-1j * corrPhaseAllPhase)
                        magDataAllPhase = magDataAllPhase[:int(numPoints / 2) + 1]
                        maxMagDataAllPhase = np.argmax(magCorrTmpDataAllPhase)
                        if maxMagDataAllPhase < max0:
                            posRamp = -1
                        incAllPhase += 1e-3
                    if incAllPhase > 0:
                        incAllPhase -= 1e-3
                    for idx in range(complexDataAllPhase.shape[1]):
                        tmp = complexDataAllPhase[idxRep, idx, :]
                        # tmp = complexDataAllPhase[:,idxPhase,idxRep]
                        tmp = tmp * np.exp(posRamp * 1j * (
                                incAllPhase * (180 / numPoints) * np.arange(1, numPoints + 1) + corrPhaseAllPhase))
                        # complexDataAllPhase[:,idxPhase,idxRep] = tmp
                        complexDataAllPhase[idxRep, idx, :] = tmp

    # ----  correct resonance frequency of unwrapped data

    for idxRep in range(nR):
        for idxPhase in range(0, complexDataAllPhaseUnw.shape[1] - numPhaseSteps + 1, numPhaseSteps):
            magDataAllPhaseUnw = abs(
                np.real(np.fft.fftshift(np.fft.fft(np.real(complexDataAllPhaseUnw[idxRep, idxPhase, :])))))
            numPointsUnw = len(complexDataAllPhaseUnw[idxRep, idxPhase, :])
            maxMagDataAllPhaseUnw = np.argmax(magDataAllPhaseUnw)
            max0 = maxMagDataAllPhaseUnw
            posRamp = 1
            incAllPhaseUnw = 0
            corrPhaseAllPhaseUnw = 0
            tmpDataAllPhaseUnw = complexDataAllPhaseUnw[idxRep, idxPhase, :]
            originalPhaseAllPhaseUnw = np.angle(tmpDataAllPhaseUnw[0])
            if w0corr:
                if maxMagDataAllPhaseUnw < (numPointsUnw // 2 + 1):
                    while maxMagDataAllPhaseUnw < (numPointsUnw // 2 + 1) or incAllPhaseUnw == 1:
                        corrTmpDataAllPhaseUnw = tmpDataAllPhaseUnw * np.exp(
                            posRamp * 1j * incAllPhaseUnw * (180 / numPointsUnw) * np.arange(1, numPointsUnw + 1))
                        magCorrTmpDataAllPhaseUnw = abs(
                            np.real(np.fft.fftshift(np.fft.fft(np.real(corrTmpDataAllPhaseUnw)))))
                        newPhaseAllPhaseUnw = np.angle(corrTmpDataAllPhaseUnw[0])
                        corrPhaseAllPhaseUnw = newPhaseAllPhaseUnw - originalPhaseAllPhaseUnw
                        corrTmpDataAllPhaseUnw = corrTmpDataAllPhaseUnw * np.exp(-1j * corrPhaseAllPhaseUnw)
                        magDataAllPhaseUnw = magDataAllPhaseUnw[:numPointsUnw // 2 + 1]
                        maxMagDataAllPhaseUnw = np.argmax(magCorrTmpDataAllPhaseUnw)
                        incAllPhaseUnw += 1e-3
                    if incAllPhaseUnw > 0:
                        incAllPhaseUnw -= 1e-3
                    tmp = complexDataAllPhaseUnw[idxRep, idxPhase:idxPhase + numPhaseSteps, :]
                    tmp = np.squeeze(tmp)
                    for idx in range(numPhaseSteps):
                        tmp[idx, :] = tmp[idx, :] * np.exp(posRamp * 1j * (
                                incAllPhaseUnw * (180 / numPointsUnw) * np.arange(1,
                                                                                  numPointsUnw + 1) + corrPhaseAllPhaseUnw))
                    complexDataAllPhaseUnw[idxRep, idxPhase:idxPhase + numPhaseSteps, :] = tmp
                elif maxMagDataAllPhaseUnw > (numPointsUnw // 2 + 1):
                    while maxMagDataAllPhaseUnw > (numPointsUnw // 2 + 1) or incAllPhaseUnw == 1:
                        corrTmpDataAllPhaseUnw = tmpDataAllPhaseUnw * np.exp(
                            posRamp * 1j * incAllPhaseUnw * (180 / numPointsUnw) * np.arange(1, numPointsUnw + 1))
                        magCorrTmpDataAllPhaseUnw = np.abs(
                            np.real(np.fft.fftshift(np.fft.fft(np.real(corrTmpDataAllPhaseUnw)))))

                        newPhaseAllPhaseUnw = np.angle(corrTmpDataAllPhaseUnw[-1])
                        corrPhaseAllPhaseUnw = newPhaseAllPhaseUnw - originalPhaseAllPhaseUnw
                        corrTmpDataAllPhaseUnw = corrTmpDataAllPhaseUnw * np.exp(-1j * corrPhaseAllPhaseUnw)
                        magDataAllPhaseUnw = magDataAllPhaseUnw[:numPointsUnw // 2 + 1]
                        maxMagDataAllPhaseUnw = np.argmax(magCorrTmpDataAllPhaseUnw)
                        incAllPhaseUnw -= 1e-3

                    if incAllPhaseUnw > 0:
                        incAllPhaseUnw += 1e-3

                    tmp = complexDataAllPhaseUnw[idxRep, idxPhase:idxPhase + (numPhaseSteps - 1), :]
                    tmp = np.squeeze(tmp)
                    for idx in range(numPhaseSteps - 1):
                        tmp[idx, :] = tmp[idx, :] * np.exp(posRamp * 1j * (
                                incAllPhaseUnw * (180 / numPointsUnw) * np.arange(1,
                                                                                  numPointsUnw + 1) + corrPhaseAllPhaseUnw))

                    complexDataAllPhaseUnw[idxRep, idxPhase:idxPhase + (numPhaseSteps - 1), :] = tmp

    # ---- 09 Compute tqFID

    realDataAllPhase = np.fft.fftshift(np.fft.fft(complexDataAllPhase, axis=-1))
    realDataAllPhaseUnw = np.fft.fftshift(np.fft.fft(complexDataAllPhaseUnw, axis=-1))
    # pos = np.argmax(np.abs(realDataAllPhaseUnw), axis=0)
    # pos = np.squeeze(pos)
    pos = int(np.floor(len(realDataAllPhaseUnw[:, 0]) / 2) + 1)

    # # Mache daraus einen Spaltenvektor
    tqFID = np.squeeze(np.mean(complexDataAllPhase.transpose(), axis=0))

    if freqDriftVal[1] != 0:
        size_tqFID = np.shape(tqFID)
        if freqDriftVal[2] != 0:
            freqDriftPerDatapoint = freqDriftVal[2]
        else:
            freqDriftPerDatapoint = (freqDriftVal[0] - method['PVM_FrqWorkOffset'][0]) / (
                    freqDriftVal[1] - acqp.get_value('ACQ_abs_time')[0])
        tqFID = tqFID * np.exp(-1j * np.arange(size_tqFID[0]) * freqDriftPerDatapoint)

    if phaseCorr:
        if tqFID.ndim < 2:
            tqFID = np.expand_dims(tqFID, axis=1)
        for idx in range(tqFID.shape[1]):
            tqFID[:, idx] = tqFID[:, idx] * np.exp(-1j * np.angle(tqFID[0, idx]))

    # ---- 10 PostDCcomp -------------------------------------------------------------------
    if postDCcomp:
        for idx in range(tqFID.shape[1]):
            tqFID[:, idx] = tqFID[:, idx] - np.mean(tqFID[-int(tqFID.shape[0] * 0.2):, idx])

    # ---- Filter FID

    if filterFID:
        timeVecCos = np.linspace(0, 0.5 * np.pi, np.fix(tqFID.shape[0] / filterFacPost).astype(int))
        for idx in range(tqFID.shape[1]):
            filterVec = np.cos(timeVecCos) ** 2
            filterVec = np.append(filterVec, np.zeros(tqFID.shape[0] - len(filterVec)))
            tqFID[:, idx] = tqFID[:, idx] * filterVec
            tqFID[len(timeVecCos):, idx] = 0
    # ---- Calculate TQ spectra

    tqSpectra = np.zeros_like(tqFID, dtype=np.complex128)
    if method['EvoTimeStep'] != 0:
        tqFID[0, :] = tqFID[0, :] * 0.5
    # RH
    for idx in range(tqFID.shape[1]):
        if onlyReal == 1:
            tqSpectra[:, idx] = np.fft.fftshift(np.fft.fft(np.squeeze(np.real(tqFID[:, idx]))))  # sym
        elif onlyReal == 0:
            tqSpectra[:, idx] = np.fft.fftshift(np.fft.fft(np.squeeze(tqFID[:, idx])))  # asym
        else:
            raise ValueError('parameter onlyReal must be either 1 (symmetric) or 0 (asymmetric)')

    if method['EvoTimeStep'] != 0:
        tqFID[0, :] = tqFID[0, :] * 2
    mixTime = method['MixTime']
    evoTime = method['EvoTime']
    return method, rawComplexData, complexDataAllPhase, complexDataUnw, realDataAllPhase, realDataAllPhaseUnw, tqFID, tqSpectra, mixTime, evoTime
