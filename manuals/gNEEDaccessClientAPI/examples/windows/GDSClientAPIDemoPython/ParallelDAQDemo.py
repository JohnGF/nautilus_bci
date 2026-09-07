import pygds
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from queue import Queue
from datetime import datetime
from threading import Thread, Condition

# define acquisition device constants (Device --> GDS)
SERIALNUMBERS = ['HA-2011.09.08', 'UA-2008.12.08']
SAMPLINGRATE = 256
BLOCKSIZE = 8       # block size for GetData (data retrieval from GDS --> Python)
                    # integral value; should be an integral multiple of 'NUMBEROFSCANS' parameter (zero needs to be treated specially)

NUMBEROFSCANS = 0   # block size for amplifier (data transmission from device --> GDS)
                    # zero for default, values depending on sampling rate

# define visualization constants
SHOWLASTXSECONDS = 1 

# define file name
FILENAME_PREFIX = 'RecordedData'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# required for dispatching plot commands to main thread
dispatchQueue = Queue()
continueDAQ = True

def dispatch(func_to_call_from_main_thread):
    dispatchQueue.put(func_to_call_from_main_thread)

# callback for each GetData cycle when the specified block size of samples was acquired
def GetDataCallback(dataBlock, filename, scope):
    global continueDAQ    
    samples = []

    # open the data file and append the latest block of data samples
    try:
        file = io.loadmat(filename, variable_names = ["data"])
        samples = file['data']
    except Exception as e:
        # Just print(e) is cleaner and more likely what you want,
        # but if you insist on printing message specifically whenever possible...
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)

    if len(samples) == 0:
        samples = np.vstack(dataBlock.copy())
    else:
        samples = np.vstack((samples, dataBlock.copy()))

    io.savemat(filename, {"data":samples})
    
    # show the last X seconds of the data
    s = len(samples)
    l = SHOWLASTXSECONDS*SAMPLINGRATE
    if (s % l == 0):
        # plotting from within a thread is not safe, so dispatch to main thread
        dispatch(lambda: scope(samples[len(samples)-min(len(samples),SHOWLASTXSECONDS*SAMPLINGRATE):]))    

    return continueDAQ

def RunDevice(deviceNr, deviceSerial, startCondition, scope):
    global continueDAQ
    
    with startCondition:
        try:
            print('Starting device', deviceSerial, '...')
            d=pygds.GDS(deviceSerial)
        except Exception as e:
            continueDAQ = False
            print('  ERROR:', e)
            return
        finally:
            # next device can be opened now
            startCondition.notify()

    try:
        d.NumberOfScans = NUMBEROFSCANS
        d.SamplingRate = SAMPLINGRATE
        d.CommonGround = [1] * 4
        d.CommonReference = [1] * 4
        d.ShortCutEnabled = 1
        d.CounterEnabled = 1
        d.TriggerEnabled = 1
        for ch in d.Channels:
            ch.Acquire = 1
            ch.BandpassFilterIndex = -1
            ch.NotchFilterIndex = -1
            ch.BipolarChannel = 0
                
        d.SetConfiguration()

        # start and run data acquisition loop (close data viewer window to stop)
        filename = FILENAME_PREFIX + '_' + timestamp + '_' + '-'+ str(deviceNr) + '.mat'
        callback = lambda samples: GetDataCallback(samples, filename, scope)
        data = d.GetData(BLOCKSIZE, callback)

        print('Stopped device', deviceSerial)
    except Exception as e:
        print('  ERROR:', e)
    finally:
        continueDAQ = False
        d.Close()
        del d

# data acquisition of each device runs in a separate thread
startCondition = Condition()

dataViewers = [pygds.Scope(1 / SAMPLINGRATE, xlabel='t/s', ylabel='U/$\mu$V', 
    title="{0} (%s channels)".format(serialNumber))
    for i, serialNumber in enumerate(SERIALNUMBERS)]

daqUnits = [Thread(target=RunDevice, args=(i,serialNumber,startCondition,
    dataViewers[i])) for i, serialNumber in enumerate(SERIALNUMBERS)]

# open devices sequentially (data acquisition will run in parallel)
with startCondition:
    for daqUnit in daqUnits:
        daqUnit.start()
        startCondition.wait()

# display loop must run in main thread
# (close a data viewer window of any device to terminate acquisition)
while continueDAQ:
    # executes the dispatched scope() commands, which return False when data 
    # viewer window is closed to indicate that data acquisition should end
    plotData = dispatchQueue.get() # blocks until an item is available
    if not plotData():
        continueDAQ = False

# close all opened data viewer windows
[plt.close(scope.fig) for scope in dataViewers]
        
# wait until the acquisition threads of all devices terminate
for daqUnit in daqUnits:
    daqUnit.join()
    daqUnits = None

print('Done.')
