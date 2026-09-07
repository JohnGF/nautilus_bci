% This is a demo script for the use of g.HIamp in the g.NEEDaccess MATLAB
% API. It records data for 10 seconds from all analog channels available
% using the internal test signal and the counter and stores the recorded
% data.

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

% set number of channels to be recorded (all available in this example)
num_of_ch_2_rec = size(find(available_channels == true),2);

% edit configuration to have a sampling rate of 256Hz, 4 scans,all
% available analog channels as well as the Counter (recorded on first
% analog channel).
% Acquire the internal test signal of g.HIamp
ghiamp_config.SamplingRate = 256;
ghiamp_config.NumberOfScans = 8;
ghiamp_config.CounterEnabled = true;
ghiamp_config.TriggerLinesEnabled = false;
ghiamp_config.HoldEnabled = false;
ghiamp_siggen = gHIampInternalSignalGenerator();
ghiamp_siggen.Enabled = true;
ghiamp_siggen.Frequency = 10;
ghiamp_config.InternalSignalGenerator = ghiamp_siggen;
for i=1:size(ghiamp_config.Channels,2)
    if (available_channels(1,i))
    	ghiamp_config.Channels(1,i).Available = true;
        if (i <= num_of_ch_2_rec)
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

% record data for 10 seconds, allocate memory for data acquired, data type
% is single
samples_acquired = 0;
data_received = single(zeros(2560, num_of_ch_2_rec));
while (samples_acquired < 2560)
    try
        [scans_received, data] = gds_interface.GetData(8);
    catch ME
        disp(ME.message);
        break;
    end
    data_received(((samples_acquired + 1) : (samples_acquired + scans_received)),:) = data;
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
clear ghiamp_config;
clear data_received;
