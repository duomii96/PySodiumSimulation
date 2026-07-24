from commonImports import *
import time
import os
"""
Taken and modified from:
https://github.com/jdoepfert/brukerMRI/blob/e70f90d95275035546cb54ec09e2f560848d9122/BrukerMRI.py#L306
"""

class BrukerData:
    """Class to store and process data of a Bruker MRI Experiment"""
    def __init__(self, path="", ExpNum=0, B0=9.4):
        self.method = {}
        self.acqp = {}
        self.reco = {}

        self.raw_fid = np.array([])
        self.proc_data = np.array([])
        self.k_data = np.array([])
        self.reco_data = np.array([])
        self.reco_data_norm = np.array([]) # normalized reco

        self.B0 = B0 # only needed for UFZ method
        self.GyroRatio = 0 # only needed for UFZ method
        self.ConvFreqsFactor = 0 # reference to convert Hz <--> ppm
        self.path = path
        self.ExpNum = ExpNum

def ReadRawData(filepath):
    with open(filepath, "r") as f:
        rawdata = np.fromfile(f, dtype=np.int32)
        f.close()
        return rawdata
def ReadParamFile(filepath):
    """
    Read a Bruker MRI experiment's method or acqp file to a
    dictionary.
    """
    param_dict = {}

    with open(filepath, "r") as f:
        lineCount = 1
        while True:
            line = f.readline()


            if not line:
                break
            lineCount += 1

            # when line contains parameter
            if line.startswith('##$'):


                (param_name, current_line) = line[3:].split('=') # split at "="

                # if current entry (current_line) is arraysize
                if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                    value = ParseArray(f, current_line)

                # if current entry (current_line) is struct/list
                elif current_line[0] == "(" and current_line[-3:-1] != " )":

                    # if neccessary read in multiple lines
                    while current_line[-2] != ")":
                        current_line = current_line[0:-1] + f.readline()

                    # parse the values to a list
                    value = [ParseSingleValue(x)
                             for x in current_line[1:-2].split(', ')]

                # otherwise current entry must be single string or number
                else:
                    value = ParseSingleValue(current_line)

                # save parsed value to dict
                param_dict[param_name] = value
            elif not(line.startswith('##$')) and line:

                continue



    return param_dict

def ParseArray(current_file, line):

    # extract the arraysize and convert it to numpy
    #print(line)
    line = line[1:-2].replace(" ", "").split(",")
    #print(line)
    arraysize = np.array([int(x) for x in line])

    # then extract the next line
    vallist = current_file.readline().split()

    # if the line was a string, then return it directly
    try:
        float(vallist[0])
    except ValueError:
        return " ".join(vallist)

    vallist = checkForAtinList(vallist) # first time check



    # include potentially multiple lines
    while len(vallist) != np.prod(arraysize):
        vallist = vallist + current_file.readline().split()

        updated = checkForAtinList(vallist) # second time check
        vallist = updated

    # try converting to int, if error, then to float
    try:
        vallist = [int(x) for x in vallist]
    except ValueError:
        vallist = [float(x) for x in vallist]

    # convert to numpy array
    if len(vallist) > 1:
        return np.reshape(np.array(vallist), arraysize)
    # or to plain number
    else:
        return vallist[0]

def ParseSingleValue(val):


    try:  # check if int
        result = int(val)
    except ValueError:
        try:  # then check if float
            result = float(val)
        except ValueError:
            # if not, should  be string. Remove  newline character.
            result = val.rstrip('\n')

    return result

def ReadExperiment(path, Exp=None, job='job0'):
    """Read in a Bruker MRI Experiment. Returns raw data, processed
    data, and method and acqp parameters in a dictionary.
    Specify path with Experiment number.
    """
    data = BrukerData(path)

    t1 = time.time()

    # parameter files
    #print(f'{path}')
    data.method = ReadParamFile(os.path.join(path, "method"))
    data.acqp = ReadParamFile(os.path.join(path, "acqp"))
    #data.reco = ReadParamFile(path + str(ExpNum) + "/pdata/1/reco")
    t2 = time.time() - t1



    # processed data
    """data.proc_data = ReadProcessedData(path + str(ExpNum) + "/pdata/1/2dseq",
                                       data.reco,
                                       data.acqp)"""

    # generate complex FID
    
    if job != 'job0':
        # for T2fast eval
        raw_data = ReadRawData(os.path.join(path, "rawdata.jobT2"))
        data.raw_fid = raw_data[0::2] + 1j * raw_data[1::2]
        t3 = time.time() -t2    
    else:
        raw_data = ReadRawData(os.path.join(path, "rawdata.job0"))
        data.raw_fid = raw_data[0::2] + 1j * raw_data[1::2]
        t3 = time.time() -t2


    # calculate GyroRatio and ConvFreqsFactor
    #data.GyroRatio = data.acqp["SFO1"]*2*np.pi/data.B0*10**6 # in rad/Ts
    #data.ConvFreqsFactor = 1/(data.GyroRatio*data.B0/10**6/2/np.pi)

    data.path = path
    #data.ExpNum = data.path[len(data.path)-1]

    return data


def checkForAtinList(valList):
    adfac= 0
    vallistTemp = valList
    for idx, item in enumerate(valList):
        newIdxStart = idx + adfac  # idx for vallistTemp to insert new method
        if '@' in item:
            singVals = str(item)
            try:
                value = int(singVals[-2])
            except:
                value = int(singVals[-3])
            mult = []
            for l in singVals[1:]:
                try:
                    mult.append(int(l))

                except:
                    print(f'EXECEption at {l}')
                    break

                fac = int("".join(map(str, mult)))
            arrayToInsert = [value] * fac
            adfac += fac - 1  # update starting index for when more lists have to be inserted
            # stick the list together again
            front = vallistTemp[:newIdxStart]
            end = vallistTemp[newIdxStart + 1:]
            # create seperate list to store data
            vallistTemp = front + arrayToInsert + end
            lenVallistTemp = len(vallistTemp)

    return vallistTemp
