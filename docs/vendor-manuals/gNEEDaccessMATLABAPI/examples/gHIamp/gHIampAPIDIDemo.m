% This is a demo script for the use of g.HIamp in the g.NEEDaccess MATLAB
% API.
% It records one channel of g.HIamp with the internal test signal and the
% digital input of g.HIamp (digital input 1) for 10 seconds and plots the
% data during acquisition in the DSP time scope.

% create time scope with 2 input channels, 256Hz sample rate and a buffer
% length of 10 seconds (2560 samples per channel)
scope_handle = dsp.TimeScope(2, 256, 'BufferLength', 2560,...
    'TimeAxisLabels', 'Bottom', 'YLimits', [-1000 1000], 'TimeSpan', 10,...
    'LayoutDimensions', [2,1], 'ReduceUpdates', true, 'YLabel', 'Amplitude [µV]');
% switch to second axes object to change limit and label
set(scope_handle, 'ActiveDisplay', 2, 'YLimits', [0 65535], 'YLabel', 'Digital Inputs');

% create gtecDeviceInterface object
gds_interface = gtecDeviceInterface();

% define connection settings (loopback)
gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.LocalPort = 50224;
gds_interface.HostPort = 50223;

% get connected devices
connected_devices = gds_interface.GetConnectedDevices();

% create g.HIamp configuration object
ghiamp_config = gHIampDeviceConfiguration();
% set serial number in g.HIamp device configuration
ghiamp_config.Name = connected_devices(1,1).Name;

% set configuration to use functions in gds interface which require device
% connection
gds_interface.DeviceConfigurations = ghiamp_config;

% get available channels
available_channels = gds_interface.GetAvailableChannels();

% edit configuration to have a sampling rate of 256Hz, 4 scans,all
% available analog channels as well as the Counter (recorded on first
% analog channel).
% Acquire the internal test signal of g.HIamp
ghiamp_config.SamplingRate = 256;
ghiamp_config.NumberOfScans = 8;
ghiamp_config.CounterEnabled = false;
ghiamp_config.TriggerLinesEnabled = true;
ghiamp_config.HoldEnabled = false;
ghiamp_siggen = gHIampInternalSignalGenerator();
ghiamp_siggen.Enabled = true;
ghiamp_siggen.Frequency = 10;
ghiamp_config.InternalSignalGenerator = ghiamp_siggen;
for i=1:size(ghiamp_config.Channels,2)
    if (available_channels(1,i))
    	ghiamp_config.Channels(1,i).Available = true;
        if (i <= 80)
            ghiamp_config.Channels(1,i).Acquire = true;
        else
            ghiamp_config.Channels(1,i).Acquire = false;
        end
        % do not use filters
        ghiamp_config.Channels(1,i).BandpassFilterIndex = -1;
        ghiamp_config.Channels(1,i).NotchFilterIndex = -1;
        % do not use a bipolar channel
        ghiamp_config.Channels(1,i).ReferenceChannel = 0;
    end
end

% apply configuration to the gds interface
gds_interface.DeviceConfigurations = ghiamp_config;
% set configuration provided in DeviceConfigurations
gds_interface.SetConfiguration();

% start data acquisition
gds_interface.StartDataAcquisition();
% start streaming
gds_interface.StartStreaming();

% record data for 10 second and plot three channels (analog channel 1,
% counter and validation indicator) of each scan acquired
samples_acquired = 0;
while (samples_acquired < 2560)
    try
        [scans_received, data] = gds_interface.GetData(8);
    catch ME
        disp(ME.message);
        break;
    end
    step(scope_handle, data(:,1),data(:,81));
    samples_acquired = samples_acquired + scans_received;
end

% stop streaming
gds_interface.StopStreaming();
% stop data acquisition
gds_interface.StopDataAcquisition();

% clean up
delete(gds_interface)

clear gds_interface;
clear ghiamp_config;
clear scope_handle;
