function datastruct = unicornrecorder_read(filename)
% UNICORNRECORDER_READ - Reads a .csv file recorded with Unicorn Recorder.
%
%   [datastruct] = UNICORNRECORDER_READ(filename) imports a
%   Unicorn Recorder data file and returns data, as well as recording
%   information.
%
%   FILENAME: string containing the name of file to import.
%   
%   DATASTRUCT: MATLAB structure containing all information stored in the
%   Unicorn Recorder data file.
%
%   (c) g.tec neurotechnology GmbH
%
%   datastruct.samplingRate
%   datastruct.data
%   datastruct.channels
%   datastruct.numberOfChannels
%   datastruct.numberOfSamples
%
%   samplingRate:
%   -------------
%   The sampling rate, data is recorded with.
%   
%   data:
%   -----
%   Recorded data is stored in the matrix.
%   Rows represent the number of samples.
%   Columns represent the number of channels.
%
%   channels:
%   ---------
%   Channel names in the order as written into the data matrix 
%   (each column represents one channel).
%
%   numberOfChannels:
%   -----------------
%   The number of channels recorded.
%
%   numberOfSamples:
%   -----------------
%   The number of samples recorded.

datastruct = unicornrecorderfileimport(filename);
end

