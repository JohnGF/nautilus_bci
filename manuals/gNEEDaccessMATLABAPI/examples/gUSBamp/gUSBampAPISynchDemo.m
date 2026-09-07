% This is a demo script for the use of g.USBamp in the g.NEEDaccess MATLAB
% API, using the time domain signal display of the DSP system toolbox of
% MATLAB.
% It records all channels of two synchronized g.USBamp with the sinusodial
% test signal for 10 seconds and displays the acquired data online in the
% DSP time scope.

% create time scope with 2 input channels (one for each amplifier), 256Hz
% sample rate and a buffer length of 10 seconds (2560 samples per channel)
scope_handle = dsp.TimeScope(2,256, 'BufferLength', 2560,...
    'YLimits', [-200000 200000], 'TimeSpan', 5, 'LayoutDimensions', [2,1],...
    'ReduceUpdates',true, 'YLabel','Amplitude [µV]');
% switch to second axes object to change limit and label
set(scope_handle, 'ActiveDisplay',2, 'YLimits', [-200000 200000], 'YLabel','Amplitude [µV]');

% create gtecDeviceInterface object
gds_interface = gtecDeviceInterface();

% define connection settings (loopback)
gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.LocalPort = 50224;
gds_interface.HostPort = 50223;

% get connected devices
connected_devices = gds_interface.GetConnectedDevices();

% create array of g.USBamp configuration objects (first configuration is
% master, second is slave
gusbamp_configs(1,1:2) = gUSBampDeviceConfiguration();
% set serial numbers in g.USBamp device configurations
% master
gusbamp_configs(1,1).Name = connected_devices(1,1).Name;
% slave
gusbamp_configs(1,2).Name = connected_devices(1,2).Name;

% set configuration to use functions in gds interface which require device
% connection
gds_interface.DeviceConfigurations = gusbamp_configs;

% get available channels
available_channels_master = gds_interface.GetAvailableChannels(connected_devices(1,1).Name);
available_channels_slave = gds_interface.GetAvailableChannels(connected_devices(1,2).Name);

% edit configuration to have a sampling rate of 256Hz, 4 scans and to
% record all 16 analog channels.
% Acquire the internal test signal of g.USBamp with a frequency of 10 Hz
% and a sine wave as wave shape. Amplitude and Offset are set to record +/-
% 200 mV. Settings are the same for both amplifiers (sampling rate and
% number of scans must be identical for both amplifiers, other settings may
% vary)
for i=1:size(gusbamp_configs,2)
    gusbamp_configs(1,i).SamplingRate = 256;
    gusbamp_configs(1,i).NumberOfScans = 8;
    gusbamp_configs(1,i).CommonGround = false(1,4);
    gusbamp_configs(1,i).CommonReference = false(1,4);
    gusbamp_configs(1,i).ShortCutEnabled = false;
    gusbamp_configs(1,i).CounterEnabled = false;
    gusbamp_configs(1,i).TriggerEnabled = false;
    gusbamp_siggen = gUSBampInternalSignalGenerator();
    gusbamp_siggen.Enabled = true;
    gusbamp_siggen.Frequency = 10;
    gusbamp_siggen.WaveShape = 3;
    gusbamp_siggen.Amplitude = 200;
    gusbamp_siggen.Offset = 0;
    gusbamp_configs(1,i).InternalSignalGenerator = gusbamp_siggen;
    for j=1:size(gusbamp_configs(1,i).Channels,2)
        if (available_channels_master(1,j))
            gusbamp_configs(1,i).Channels(1,j).Available = true;
            gusbamp_configs(1,i).Channels(1,j).Acquire = true;
            % do not use filters
            gusbamp_configs(1,i).Channels(1,j).BandpassFilterIndex = -1;
            gusbamp_configs(1,i).Channels(1,j).NotchFilterIndex = -1;
            % do not use a bipolar channel
            gusbamp_configs(1,i).Channels(1,j).BipolarChannel = 0;
        end
    end
end

% apply configuration to the gds interface
gds_interface.DeviceConfigurations = gusbamp_configs;
% set configuration provided in DeviceConfigurations
gds_interface.SetConfiguration();

% start data acquisition
gds_interface.StartDataAcquisition();
% start streaming
gds_interface.StartStreaming();

% record data for 10 second and plot the first analog channel acquired.
samples_acquired = 0;
while (samples_acquired < 2560)
    try
        [scans_received, data] = gds_interface.GetData(8);
    catch ME
        disp(ME.message);
        break;
    end
    % display first analog channel for each device in time scope
    step(scope_handle, data(:,1), data(:,17));
    samples_acquired = samples_acquired + scans_received;
end

% stop streaming
gds_interface.StopStreaming();
% stop data acquisition
gds_interface.StopDataAcquisition();

% clean up
delete(gds_interface)

scope_handle.hide;

clear gds_interface;
clear gusbamp_config;
clear scope_handle;
