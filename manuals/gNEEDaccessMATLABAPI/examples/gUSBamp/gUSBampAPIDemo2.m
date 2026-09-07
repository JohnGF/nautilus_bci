% This is a demo script for the use of g.USBamp in the g.NEEDaccess MATLAB
% API.
% It records all channels of g.USBamp with the sinusodial internal test
% signal (+/- 200mV, 10Hz) for 10 seconds and saves the acquired data after
% recording.

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
% record all 16 analog channels.
gusbamp_config.SamplingRate = 256;
gusbamp_config.NumberOfScans = 8;
gusbamp_config.CommonGround = false(1,4);
gusbamp_config.CommonReference = false(1,4);
gusbamp_config.ShortCutEnabled = false;
gusbamp_config.CounterEnabled = false;
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

% start data acquisition
gds_interface.StartDataAcquisition();
% start streaming
gds_interface.StartStreaming();

% record data for 10 second and plot the first analog channel acquired.
samples_acquired = 0;
data_received = single(zeros(2560, 16));
while (samples_acquired < 2560)
    try
        [scans_received, data] = gds_interface.GetData(8);
        
        data_received((samples_acquired + 1) : (samples_acquired + scans_received), :) = data;
    catch ME
        disp(ME.message);
        break;
    end
    samples_acquired = samples_acquired + scans_received;
end

% stop streaming
gds_interface.StopStreaming();
% stop data acquisition
gds_interface.StopDataAcquisition();

% clean up
delete(gds_interface)

% get user directory to save data in Documents/MATLAB folder
user_profile = getenv('USERPROFILE');
dirname = sprintf('%s\\Documents\\MATLAB', user_profile);
filename = sprintf('%s\\data_received.mat', dirname);
% convert data to double for later use in g.BSanalyze
data_received = double(data_received);
% if folder exists save variable there, if not do not save
if (exist(dirname,'dir') == 7)
    save(filename, 'data_received');
end

clear gds_interface;
clear gusbamp_config;
clear data_received;
