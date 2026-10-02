% This is a demo script for the use of g.Nautilus in the g.NEEDaccess
% MATLAB API.
% It records data for 10 seconds from all analog channels available and
% the digital inputs of g.Nautilus. Three recorded channels (first analog
% channel, digital inputs and validation indicator) are plotted online
% during data acquisition.

% create time scope with 3 input channels, 250Hz sample rate and a buffer
% length of 10 seconds (2500 samples per channel)
scope_handle = dsp.TimeScope(3, 250, 'BufferLength', 2500,...
    'TimeAxisLabels', 'Bottom', 'YLimits', [0 16000], 'TimeSpan', 10,...
    'LayoutDimensions', [3,1],  'ReduceUpdates', true,...
    'YLabel', 'Amplitude [µV]');
% switch to second axes object to change limit and label
set(scope_handle, 'ActiveDisplay', 2, 'YLimits', [0 255], 'YLabel', 'Digital Inputs');
% switch to third axes object to change limit and label
set(scope_handle, 'ActiveDisplay', 3, 'YLimits', [0 1], 'YLabel', 'Valid');

% create gtecDeviceInterface object
gds_interface = gtecDeviceInterface();

% define connection settings (loopback)
gds_interface.IPAddressHost = '127.0.0.1';
gds_interface.IPAddressLocal = '127.0.0.1';
gds_interface.LocalPort = 50224;
gds_interface.HostPort = 50223;

% get connected devices
connected_devices = gds_interface.GetConnectedDevices();

% create g.Nautilus configuration object
gnautilus_config = gNautilusDeviceConfiguration();
% set serial number in g.Nautilus device configuration
gnautilus_config.Name = connected_devices(1,1).Name;

% set configuration to use functions in gds interface which require device
% connection
gds_interface.DeviceConfigurations = gnautilus_config;

% get network channel selected by g.Nautilus
gnautilus_config.NetworkChannel = gds_interface.GetNetworkChannel();

% get available channels
available_channels = gds_interface.GetAvailableChannels();
% get supported sensitivities
supported_sensitivities = gds_interface.GetSupportedSensitivities();
% get supported input sources
supported_input_sources = gds_interface.GetSupportedInputSources();

% edit configuration to have a sampling rate of 250Hz, 4 scans, all
% available analog channels as well as ValidationIndicator and DigitalIOs.
% Acquire the internal test signal of g.Nautilus
gnautilus_config.SamplingRate = 250;
gnautilus_config.NumberOfScans = 8;
gnautilus_config.InputSignal = supported_input_sources(3).Value;
gnautilus_config.NoiseReduction = false;
gnautilus_config.CAR = false;
% acquire additional channels digital inputs and validation indicator
gnautilus_config.DigitalIOs = true;
gnautilus_config.ValidationIndicator = true;
% do not acquire other additional channels
gnautilus_config.AccelerationData = false;
gnautilus_config.LinkQualityInformation = false;
gnautilus_config.BatteryLevel = false;
gnautilus_config.Counter = false;
for i=1:size(gnautilus_config.Channels,2)
    if (available_channels(1,i))
    	gnautilus_config.Channels(1,i).Available = true;
        gnautilus_config.Channels(1,i).Acquire = true;
        % set sensitivity to 187.5 mV
        gnautilus_config.Channels(1,i).Sensitivity = supported_sensitivities(6);
        % do not use channel for CAR and noise reduction
        gnautilus_config.Channels(1,i).UsedForNoiseReduction = false;
        gnautilus_config.Channels(1,i).UsedForCAR = false;
        % do not use filters
        gnautilus_config.Channels(1,i).BandpassFilterIndex = -1;
        gnautilus_config.Channels(1,i).NotchFilterIndex = -1;
        % do not use a bipolar channel
        gnautilus_config.Channels(1,i).BipolarChannel = -1;
    end
end

% apply configuration to the gds interface
gds_interface.DeviceConfigurations = gnautilus_config;
% set configuration provided in DeviceConfigurations
gds_interface.SetConfiguration();

% start data acquisition
gds_interface.StartDataAcquisition();
% start streaming
gds_interface.StartStreaming();

% record data for 10 second and plot three channels (analog channel 1,
% counter and validation indicator) of each scan acquired
samples_acquired = 0;
while (samples_acquired < 2500)
    try
        [scans_received, data] = gds_interface.GetData(8);
    catch ME
        disp(ME.message);
        break;
    end
    step(scope_handle, data(:,1),data(:,33),data(:,34));
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
clear gnautilus_config;
