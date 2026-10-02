% This is a demo script for the use of g.USBamp in the g.NEEDaccess MATLAB
% API.
% It records one channel of g.USBamp with a sinusodial input signal
% (internal test signal, +/- 200mV, 10Hz) and the digital inputs of
% g.USBamp for 10 seconds and stores the acquired data after recording.

% create time scope with 2 input channels, 256Hz sample rate and a buffer
% length of 10 seconds (2560 samples per channel)
scope_handle = dsp.TimeScope(2, 256, 'BufferLength', 2560,...
    'TimeAxisLabels', 'Bottom', 'YLimits', [-200000 200000],...
    'TimeSpan', 10, 'LayoutDimensions', [2,1], 'ReduceUpdates', true,...
    'YLabel', 'Amplitude [µV]');
% switch to second axes object to change limit and label
set(scope_handle, 'ActiveDisplay', 2, 'YLimits', [0 255], 'YLabel', 'Digital Inputs');

% create gtecDeviceInterface object
gds_interface = gtecDeviceInterface();

% define connection settings (loopback)
gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.LocalPort = 50224;
gds_interface.HostPort = 50223;

% get connected devices
connected_devices = gds_interface.GetConnectedDevices();

% create g.USBamp configuration object
gusbamp_config = gUSBampDeviceConfiguration();
% set serial number in g.USBamp device configuration
gusbamp_config.Name = connected_devices(1,1).Name;

% set configuration to use functions in gds interface which require device
% connection
gds_interface.DeviceConfigurations = gusbamp_config;

% get available channels
available_channels = gds_interface.GetAvailableChannels();

% edit configuration to have a sampling rate of 256Hz, 8 scans and to
% record 1 analog channel.
gusbamp_config.SamplingRate = 256;
gusbamp_config.NumberOfScans = 8;
gusbamp_config.CommonGround = false(1,4);
gusbamp_config.CommonReference = false(1,4);
gusbamp_config.ShortCutEnabled = false;
gusbamp_config.CounterEnabled = false;
gusbamp_config.TriggerEnabled = true;
gusbamp_siggen = gUSBampInternalSignalGenerator();
gusbamp_siggen.Enabled = true;
gusbamp_siggen.Frequency = 10;
gusbamp_siggen.WaveShape = 3;
gusbamp_siggen.Amplitude = 200;
gusbamp_siggen.Offset = 0;
gusbamp_config.InternalSignalGenerator = gusbamp_siggen;
for i=1:size(gusbamp_config.Channels,2)
    if (available_channels(1,i))
        if (i == 1)
            gusbamp_config.Channels(1,i).Available = true;
            gusbamp_config.Channels(1,i).Acquire = true;
        else
            gusbamp_config.Channels(1,i).Available = true;
            gusbamp_config.Channels(1,i).Acquire = false;
        end
        % do not use filters
        gusbamp_config.Channels(1,i).BandpassFilterIndex = -1;
        gusbamp_config.Channels(1,i).NotchFilterIndex = -1;
        % do not use a bipolar channel
        gusbamp_config.Channels(1,i).BipolarChannel = 0;
    end
end

% apply configuration to the gds interface
gds_interface.DeviceConfigurations = gusbamp_config;
% set configuration provided in DeviceConfigurations
gds_interface.SetConfiguration();

% start data acquisition
gds_interface.StartDataAcquisition();
% start streaming
gds_interface.StartStreaming();

% record data for 10 seconds
samples_acquired = 0;
while (samples_acquired < 2560)
    try
        [scans_received, data] = gds_interface.GetData(8);
    catch ME
        disp(ME.message);
        break;
    end
    step(scope_handle, data(:,1), data(:,2));
    samples_acquired = samples_acquired + scans_received;
end

% stop streaming
gds_interface.StopStreaming();
% stop data acquisition
gds_interface.StopDataAcquisition();

% close scope
scope_handle.hide;

% clean up
delete(gds_interface)

clear scope_handle;
clear gds_interface;
clear gusbamp_config;
