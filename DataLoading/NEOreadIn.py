import numpy as np

from commonImports import *
import PythonBrukerReadIn as pbr
from scipy.optimize import minimize
import re


def NEOreadInSE(pathToData, job='job0'):

    spikeComp = 0
    filterFID = 0
    filterFac = 1
    p = 0 # when using phosphor measurements
    filterMz = 0
    filtertyp = 0  #(1 lorentz, 0 gauss)
    sigma = 0.75
    TPMa1 = 4.2
    TPMb1 = 3.0
    usePhaseCorr = 0 # not needed for Single Pulse (oder?)

    # Load method and acqp parameters
    data_path = pathToData
    ds = pbr.ReadExperiment(path=data_path, job=job)

    rawComplexData = ds.raw_fid
    acqp = ds.acqp
    method = ds.method
    
    print(f'Method: {method["Method"]}')


    # Check if the parameters are properly loaded
    #print(method)
    #print(acqp)
    #complexData = rawComplexData[0]  # Assuming rawdata is a list or array-like structure

    if np.ndim(rawComplexData) == 1:
        complexData = np.expand_dims(rawComplexData, axis=0)
    else:
        complexData = rawComplexData

    #sizeCD = np.shape(complexData)

    # Extract relevant information
    specDim = method['PVM_SpecMatrix']
    specDim1 = 2 ** np.ceil(np.log2(method['PVM_SpecMatrix']))
    
    if "T2f" in method['Method']:
        # read acq-points from acqp["ACQ_jobs"] using regex
        input_string = acqp["ACQ_jobs"]
        match = re.search(r'\(\d+.*?\)\s*\((\d+)', input_string)
        if match:
                complexPoints = int(match.group(1))
                specDim = int(complexPoints / 2)
                complexData = np.reshape(complexData, complexData.shape[-1] // specDim[0], specDim)
        else:
            print("Value not found")
        
    else:   
        try:
            if specDim.size != specDim1.size:
                complexData = complexData[:specDim]
            elif specDim[0] != complexData.shape[-1] and specDim.size == specDim1.size:
                complexData = np.reshape(complexData, (complexData.shape[-1] // specDim, specDim))
        except:
            numEchoes = method['NumInvTime']
            complexData = np.reshape(complexData, (numEchoes, specDim[0]))
    if method['Method'] == '<User:sr_SP>' and np.min(real(complexData)) < -1e6:
        print("Data multiplied by -1 to switch it around")
        complexData *= -1e0




    return complexData, method, acqp
