import pygds
import numpy                            #requires numpy 1.15.0 or later
import math
import matplotlib.pyplot as pyplot
from timeit import default_timer

# GDS remote features (enabled by default)                  
# disable explicitly to fetch data blocks in less than 20ms intervals
disableRemoteFeatures = False

# define acquisition constants for GetData (GDS --> Python)
secondsToAcquire = 5
userBlockSize = 8

# define acquisition constants for device (g.tec amplifier --> GDS)
serialNumber = 'HA-2011.09.08'
samplingRate = 256      # g.Nautilus: 250 or 500; g.HIamp/g.USBamp: 256, 512,... and others
deviceBlockSize = 0     # zero for default values depending on sampling rate
                        # integral value; should be at least samplingRate * 0.03 sec (if remote is enabled)
                        
# logging from within the receiveDataCallback affects time measurement massively at higher sampling rates
verboseCallbackTiming = (samplingRate/userBlockSize < 1000)   

"""
if the 'deviceBlockSize' parameter used by the device (d.NumberOfScans) is not
an integral multiple of the 'userBlockSize' parameter used in d.GetData(userBlockSize,...),
timing of acquisition may only SEEM to be inaccurate:
- device delivers data in blocks to GDS service, each of them containing a number of 'deviceBlockSize' scans
- GDS service stores data into buffer
- user retrieves data from GDS buffer (not directly from the device) with d.GetData(userBlockSize, retrieveDataCallback), where 'userBlockSize' is the number of samples to retrieve and 'retrieveDataCallback' is the callback function that is called after an amount of 'userBlockSize' samples have been acquired
- if 'deviceBlockSize' is not an integral multiple of 'userBlockSize', the user might need to wait until the device delivers one more block to the GDS service until GetData can return
- this would result in a different timing for the calls to the 'retrieveDataCallback' callback function, depending on the number of samples that remain in the buffer since the last call
"""

# ---------------------------------------------------------------------

# returns a timestamp that can be 
def timestamp():
    return default_timer()

# returns the number of milliseconds elapsed between the two provided timestamps
def elapsedMilliseconds(startTime, stopTime):
    dt = (stopTime - startTime) * 1000.0
    return dt;

# callback for each GetData cycle when the specified block size of samples was acquired
def retrieveDataCallback(block):
    global sampleBlocks, nrBlocks, acquiredSamples, start, avgBlockTime, blockTimes
    time = timestamp()
    if not hasattr(retrieveDataCallback, "lastTime"):
        retrieveDataCallback.firstTime = time
        retrieveDataCallback.lastTime = start
        avgBlockTime = 0
    blockTime = elapsedMilliseconds(retrieveDataCallback.lastTime, time)
    blockTimes.append(blockTime)
    totalTime = elapsedMilliseconds(start, time)
    if (retrieveDataCallback.firstTime < time):
        avgBlockTime += blockTime / (nrBlocks - 1)
    retrieveDataCallback.lastTime = time
    success = sampleBlocks.append(block.copy())
    acquiredSamples += block.shape[0]
    if (verboseCallbackTiming): # writing to console here affects time measurement tremendously at higher sampling rates
        print(f"{acquiredSamples} samples:\t{totalTime:.3f} ms" + (f"\t(dt = {blockTime:.3f} ms)" if len(sampleBlocks) > 1 else "\t(first block includes start of acquisition)"))
    return success or len(sampleBlocks) < nrBlocks

# ---------------------------------------------------------------------

# configure GDS remote features (enabled by default; changing requires Uninitialize/Initialize)
# disabling remote features allows to fetch data blocks in less than 20 ms intervals on local machine only
print('Configuring GDS for {0} usage ({1} remote features)...'.format('local-only' if disableRemoteFeatures else 'default' , 'disabling' if disableRemoteFeatures else 'enabling'))
pygds.Uninitialize()
pygds.Initialize(gds_dll = pygds.gds_dll_standalone if disableRemoteFeatures else pygds.gds_dll_client)

sampleBlocks = []
blockTimes = []
acquiredSamples = 0
nrBlocks = secondsToAcquire * samplingRate / userBlockSize
avgBlockTime = 0

# open device
d = pygds.GDS(serialNumber)

try:
    # adjust configuration
    d.SamplingRate = samplingRate

    # set block size for acquisition from device (this is not the same as the userBlockSize for the GetData option)
    # (see detailed comment above)
    if deviceBlockSize == 0:
        # determine value for d.NumberOfScans parameter automatically
        # defaults for g.Nautilus are 8 (@250 Hz), and 15 (@500 Hz)
        d.NumberOfScans_calc()
    else:
        d.NumberOfScans = deviceBlockSize
    
    d.HoldEnabled = 1
    d.Counter = 1
    d.Trigger = 0
    for ch in d.Channels:
        ch.Acquire = 1
        ch.BandpassFilterIndex = -1
        ch.NotchFilterIndex = -1
        ch.ReferenceChannel = 0
            
    d.SetConfiguration()
    deviceBlockSize = d.NumberOfScans   # in case device block size was chosen automatically

    print("Acquiring data...")

    # acquisition (with time measurement)              
    start = timestamp()
    data = d.GetData(userBlockSize, retrieveDataCallback)
    stop = timestamp()

finally:
    icounter = d.IndexAfter('Counter')-1
    d.Close(); del d

# do we have an integral multiple of acquired device block size (d.NumberOfScans) and requested user block size (GetData)? 
# (if deviceBlockSize > userBlockSize: how many scans of the (only partially used) last block do we need for a single GetData call?)
# (if userBlockSize < deviceBlockSize: how many scans of one acquired device block do we need for a single GetData call?)   
partialBlockSize = userBlockSize % deviceBlockSize
print()

if partialBlockSize > 0:
    print(f"--> irregular GetData intervals: device's block size ({deviceBlockSize}) is not an integral multiple of GetData's user block size parameter ({userBlockSize})")

    # how many of those additional, only partially consumed device blocks do we need to consume an integral number of device blocks entirely?
    # (this also works if userBlockSize < deviceBlockSize)
    integralBlockSize = numpy.lcm(deviceBlockSize, partialBlockSize)        # number of scans that make up the integral number of device blocks
    userBlocksRequired = int(integralBlockSize / partialBlockSize)          # how many GetData calls do we need to acquire that integral number of additional (partially used) device blocks?
    regularAcquiredUserBlocks = int(integralBlockSize / deviceBlockSize)    # how many of them GetData calls must acquire that additional (partially used) block before they can deliver the requested user block size?
    
    # block durations
    deviceBlockDurationMs = 1000 * deviceBlockSize / samplingRate       # the amount of data in one device block expressed in milliseconds  
    deviceBlocksPerGetData = userBlockSize / deviceBlockSize            # the exact amount of device blocks per GetData call         
    regularBlocks = math.ceil(deviceBlocksPerGetData)                   # the integral number of device blocks to acquire regularely for a single GetData call
    intermediateBlocks = math.floor(deviceBlocksPerGetData)             # the integral number of device blocks to acquire for GetData calls that can consume the remainder from the regular ones 

    print(f"   {regularAcquiredUserBlocks} of {userBlocksRequired} GetData calls must acquire an additional block of samples from the device")
    print(f"   {regularAcquiredUserBlocks} of {userBlocksRequired} GetData calls acquire:\t{regularBlocks} device blocks   {deviceBlockSize} samples from device\t({regularBlocks*deviceBlockDurationMs:.3f} ms of data)")
    print(f"   {userBlocksRequired-regularAcquiredUserBlocks} of {userBlocksRequired} GetData calls acquire:\t{intermediateBlocks} device blocks   {deviceBlockSize} samples from device\t({intermediateBlocks*deviceBlockDurationMs:.3f} ms of data)")
else:
    print(f"--> regular GetData intervals: device's block size ({deviceBlockSize}) is an integral multiple of GetData's user block size parameter ({userBlockSize})")
    
# output elapsed time and length of samples
daqNettoDuration = elapsedMilliseconds(retrieveDataCallback.firstTime, retrieveDataCallback.lastTime)
daqTotalDuration = elapsedMilliseconds(start, stop)

print()
print(f"Acquired {acquiredSamples} total samples in {daqTotalDuration:.3f} ms (including start/stop acquisition)")
print(f"Acquired {acquiredSamples-userBlockSize} samples in {daqNettoDuration:.3f} ms (net)")
print(f"Average sample block time: {avgBlockTime:.3f} ms")
      
# plot counter and acquisition block times
if acquiredSamples > 0:
    counterScope = pygds.Scope(1/samplingRate, modal=True, ylabel='n', xlabel='t/s', title='Counter')
    counterScope(numpy.concatenate(sampleBlocks)[:, icounter:icounter+1])
    
    timingScope = pygds.Scope(1/samplingRate, modal=True, ylabel='block interval/ms', xlabel='t/s', title='Block Timing')
    timingScope(numpy.array(blockTimes).reshape(-1,1))
    
    pyplot.show()
