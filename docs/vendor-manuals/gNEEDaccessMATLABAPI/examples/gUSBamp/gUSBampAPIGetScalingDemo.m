% This is a MATLAB script for the performance test of g.NEEDaccess MATLAB
% API. It tests the g.USBamp function GetScaling.

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

% Set up a random device configuration for g.USBamp and configure device
gusbamp_config.SamplingRate = 256;
gusbamp_config.NumberOfScans = 8;
gusbamp_config.CommonGround = false(1,4);
gusbamp_config.CommonReference = false(1,4);
gusbamp_config.ShortCutEnabled = false;
gusbamp_config.CounterEnabled = true;
gusbamp_config.TriggerEnabled = false;
gusbamp_siggen = gUSBampInternalSignalGenerator();
gusbamp_siggen.Enabled = true;
gusbamp_siggen.Frequency = 10;
gusbamp_siggen.WaveShape = 3;
gusbamp_siggen.Amplitude = 200;
gusbamp_siggen.Offset = 0;
gusbamp_config.InternalSignalGenerator = gusbamp_siggen;
for i=1:size(gusbamp_config.Channels,2)
    if (available_channels(1,i))
    	gusbamp_config.Channels(1,i).Available = true;
        gusbamp_config.Channels(1,i).Acquire = true;
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

% get scaling of connected g.USBamp
current_scaling = gds_interface.GetScaling(gusbamp_config.Name);
