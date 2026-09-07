% gHIampAPIImpedanceDemo is a MATLAB script demonstrating how impedance
% measurement works in g.NEEDaccess MATLAB API using g.HIamp.

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

% Set up a random device configuration for g.HIamp and configure device
ghiamp_config.SamplingRate = 256;
ghiamp_config.NumberOfScans = 8;
ghiamp_config.CounterEnabled = true;
ghiamp_config.TriggerLinesEnabled = false;
ghiamp_config.HoldEnabled = false;
ghiamp_siggen = gHIampInternalSignalGenerator();
ghiamp_siggen.Enabled = true;
ghiamp_siggen.Frequency = 10;
ghiamp_config.InternalSignalGenerator = ghiamp_siggen;
channel_selected = zeros(1,size(available_channels(available_channels == true), 2));
for i=1:size(ghiamp_config.Channels,2)
    if (available_channels(1,i))
    	ghiamp_config.Channels(1,i).Available = true;
        if (i <= 80)
            ghiamp_config.Channels(1,i).Acquire = true;
            channel_selected(1,i) = i;
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

% return impedances for selected channels. The variable returned contains a
% 2 x channels array representing measured channel and corresponding
% impedance
impedances = gds_interface.GetImpedance(channel_selected, 'active');
