//COPYRIGHT © 2013 G.TEC MEDICAL ENGINEERING GMBH, AUSTRIA
#include "stdafx.h"
#include <Windows.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>

#include <GDSClientAPI_gUSBamp.h>

#define CONSOLE_ESCAPE_CODE_CLEAR_TO_THE_LEFT "\0"
#define CONSOLE_ESCAPE_CODE_CARRIAGE_RETURN "\r"

// Specifications for the data acquisition.
//-------------------------------------------------------------------------------------
#define DATA_FILE "data.bin"
#define SAMPLE_RATE 512 // [Hz]
#define DURATION_DAQ 20 // [s]
#define DATA_READY_TRESHOLD_MS 30 // [ms]

// Definition of network specific stuff.
//-------------------------------------------------------------------------------------
#define HOST_IP "127.0.0.1" // Default address is the loopback address, else the ip of the computer running GDS.
#define HOST_PORT 50223     // The default port of GDS is 50223.
#define LOCAL_IP "127.0.0.1"// Default address is the loppback address, else the ip of the client machine.
#define LOCAL_PORT 50224    // Any free port on the local machine.

// Function declarations.
//-------------------------------------------------------------------------------------
void setup_config( GDS_GUSBAMP_CONFIGURATION* cfg, unsigned int sample_rate );
void show_config( GDS_GUSBAMP_CONFIGURATION* cfg );

void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData);
void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void*);
void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData);

// Global events plus timeout.
//-------------------------------------------------------------------------------------
#define SYSTEM_EVENT_TIMEOUT 5000 // [ms]
HANDLE glb_event_handle;

//-------------------------------------------------------------------------------------
// main. Program entry point.
//-------------------------------------------------------------------------------------
int _tmain(int argc, _TCHAR* argv[])
{
	std::cout << "g.NEEDaccess: g.USBamp Demo" << std::endl << std::endl;

	// init global handles
	glb_event_handle = NULL;

	// Create the event which is triggerd when data should be processed.
	//-------------------------------------------------------------------------------------
	glb_event_handle = CreateEvent(NULL, false, false, NULL);
	if ( glb_event_handle == NULL )
	{
		std::cerr << "ERROR: CreateEvent failed with error code " << GetLastError() << "." << std::endl;
		std::cin.get();
		return -1;
	}

	// Setup network addresses.
	//-------------------------------------------------------------------------------------
	static const std::string host_ip(HOST_IP);
	static const uint16_t host_port = HOST_PORT;
	static const std::string local_ip(LOCAL_IP);
	static const uint16_t local_port = LOCAL_PORT;
	GDS_ENDPOINT host_endpoint, local_endpoint;
	
	strcpy( host_endpoint.IpAddress, host_ip.c_str() );
	host_endpoint.Port = host_port;
	strcpy( local_endpoint.IpAddress, local_ip.c_str() );
	local_endpoint.Port = local_port;

	// Initialize the library.
	//-------------------------------------------------------------------------------------
	GDS_Initialize();

	// Get all connected devices.
	//-------------------------------------------------------------------------------------
	GDS_DEVICE_CONNECTION_INFO* connected_devices = NULL;
	size_t count_daq_units = 0;
	GDS_RESULT ret = GDS_GetConnectedDevices( host_endpoint, local_endpoint, &connected_devices, &count_daq_units );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_GetConnectedDevices: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Show all devices of interest (i.e. g.USBamp's).
	//-------------------------------------------------------------------------------------
	std::vector<std::string> device_list;
	size_t count_devices = 0;
	// The GDS_GetConnectedDevices returns the number of daq units. The first "for" loop
	// addresses the daq units.
	for ( size_t i = 0; i < count_daq_units; i++ )
	{
		// Each daq unit operates a different number of devices. The second "for" loop 
		// addresses the devices attached to the daq unit.
		for ( size_t j  = 0; j < connected_devices[i].ConnectedDevicesLength; j++ )
		{
			if ( connected_devices[i].ConnectedDevices[j].DeviceType == GDS_DEVICE_TYPE_GUSBAMP )
			{
				if ( connected_devices[i].InUse == FALSE )
				{
					count_devices++;
					device_list.push_back( connected_devices[i].ConnectedDevices[j].Name ); 
					std::cout << "Device #" << count_devices << " (" << device_list[count_devices-1]  << " @ " << host_ip << ":" << host_port << ")" << std::endl;
				}
				else
					std::cout << "Device " <<	connected_devices[i].ConnectedDevices[j].Name  << " @ " << host_ip << ":" << host_port << " is not available " << std::endl;
			}
		}
	}
	if ( count_devices == 0 )
	{
		std::cout << "No device available, press <return> to exit" << std::endl;
		std::cin.get();
		return 0;
	}
	else
	{
		// Free the list of found devices. This list is not needed any more.
		//-------------------------------------------------------------------------------------
		ret = GDS_FreeConnectedDevicesList( &connected_devices, count_daq_units);
		if ( ret.ErrorCode != GDS_ERROR_SUCCESS )
		{
			std::cerr << "ERROR on GDS_FreeConnectedDevicesList: " << ret.ErrorMessage << std::endl;
			std::cin.get();
			return -1;
		}
	}

	// Read the input (i.e. the number of the device of interest) from the console.
	//-------------------------------------------------------------------------------------
	std::string str_selection;
	std::cout << "Select device #";
	std::getline( std::cin, str_selection );
	size_t idx_selection = atoi(str_selection.c_str())-1;
	if ( idx_selection > device_list.size()-1 )
	{
		std::cerr << "ERROR: The given device number is not valid" << std::endl;
		std::cin.get();
		return -1;
	}

	// Try to connect to the device. This example deals with only a single device.
	//-------------------------------------------------------------------------------------
	char (*device)[DEVICE_NAME_LENGTH_MAX] = new char[1][DEVICE_NAME_LENGTH_MAX];
	std::strcpy( device[0], device_list[idx_selection].c_str() );
	GDS_HANDLE connectionHandle = 0;
	BOOL is_creator = FALSE;
	ret = GDS_Connect( host_endpoint, local_endpoint, device, 1, TRUE, &connectionHandle, &is_creator );
	delete[] device;
	device = NULL;
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_Connect: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Set the callback function which will be called if the server triggers a
	// daq error event.
	//-------------------------------------------------------------------------------------
	std::string usrData = device_list[idx_selection];
	ret = GDS_SetDataAcquisitionErrorCallback(connectionHandle, on_data_acquisition_error, (void*) &usrData);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetDataAcquisitionErrorCallback: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}
 
	// Set the callback function which will be called if the server dies or is going to
	// shutdown.
	//-------------------------------------------------------------------------------------
	ret = GDS_SetServerDiedCallback( connectionHandle, on_server_died_event, NULL );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on  GDS_SetServerDiedCallbackk: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Setup a configuration.
	//-------------------------------------------------------------------------------------
	GDS_GUSBAMP_CONFIGURATION* cfg_usbamp = new GDS_GUSBAMP_CONFIGURATION;
	setup_config( cfg_usbamp, SAMPLE_RATE );
	GDS_CONFIGURATION_BASE* cfg = new GDS_CONFIGURATION_BASE[1];
	cfg[0].DeviceInfo.DeviceType = GDS_DEVICE_TYPE_GUSBAMP;
	strcpy( cfg[0].DeviceInfo.Name, device_list[idx_selection].c_str() );
	cfg[0].Configuration = cfg_usbamp;

	// Try to apply the configuration on the server.
	//-------------------------------------------------------------------------------------
	ret = GDS_SetConfiguration( connectionHandle, cfg, 1 );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetConfiguration: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Free the resources used by the config.
	//-------------------------------------------------------------------------------------
	delete cfg_usbamp;
	delete [] cfg;

	// Retrieve the configuration which has been applied to the server before.
	//-------------------------------------------------------------------------------------
	GDS_CONFIGURATION_BASE* cfg_received = NULL;
	size_t count_cfg_received = 0;
	ret = GDS_GetConfiguration( connectionHandle, &cfg_received, &count_cfg_received );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_GetConfiguration: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Write the retrieved config to the console.
	//-------------------------------------------------------------------------------------
	GDS_GUSBAMP_CONFIGURATION* cfg_received_usbamp = (GDS_GUSBAMP_CONFIGURATION*)(cfg_received[0].Configuration);
	std::cout << "Config of " << device_list[0] << ": " << std::endl;
	show_config( cfg_received_usbamp );

	// Set the data ready callback treshold.
	//-------------------------------------------------------------------------------------
	size_t data_ready_treshold_number_of_scans = std::max<size_t>( (size_t) std::floor((double)cfg_received_usbamp->SampleRate * DATA_READY_TRESHOLD_MS/1000), cfg_received_usbamp->NumberOfScans );
	ret = GDS_SetDataReadyCallback(connectionHandle, on_data_ready_event, data_ready_treshold_number_of_scans, NULL);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_SetDataReadyCallback: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Free the resources used by the config.
	//-------------------------------------------------------------------------------------
	ret = GDS_FreeConfigurationList( &cfg_received, count_cfg_received );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_FreeConfigurationList: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Retrieve the buffer size for a single scan.
	//-------------------------------------------------------------------------------------
	size_t scan_count = 1;
	size_t channels_per_device_count = 0;
	size_t buffer_size_per_scan = 0;
	ret = GDS_GetDataInfo( connectionHandle, &scan_count, NULL, &channels_per_device_count, &buffer_size_per_scan );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on 1st GDS_GetDataInfo: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// The memory for the channels per device array is allocated here (based on the first call
	// of the method). GDS_GetDataInfo fills the array with the
	// available number of channels for each device. This is done for demo reasons.
	//-------------------------------------------------------------------------------------
	size_t* channels_per_device = new size_t[channels_per_device_count];
	ret = GDS_GetDataInfo( connectionHandle, &scan_count, channels_per_device, &channels_per_device_count, &buffer_size_per_scan );

	// We do not need the information because we know our own setup.
	//-------------------------------------------------------------------------------------
	delete[] channels_per_device;
	channels_per_device = NULL;
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on 2nd GDS_GetDataInfo: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Command the device to acquire data.
	//-------------------------------------------------------------------------------------
	ret = GDS_StartAcquisition( connectionHandle );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StartAcquisition(...): " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Stream the measurement data via the network.
	//-------------------------------------------------------------------------------------
	ret = GDS_StartStreaming( connectionHandle );
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StartStreaming: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Prepare a buffer and acquire the measurement data. Save the data to the file.
	//-------------------------------------------------------------------------------------
	size_t buffer_size_in_samples = SAMPLE_RATE * buffer_size_per_scan;
	uint64_t total_scans_to_acquire = DURATION_DAQ * SAMPLE_RATE;
	uint64_t total_acquired_scans = 0;

	try
	{
		float* data_buffer = new float[buffer_size_in_samples];
		std::cout << "Start acquiring measurement data for " << DURATION_DAQ << "s." << std::endl;
		std::ofstream file( DATA_FILE, std::ios_base::binary );

		while ( total_acquired_scans < total_scans_to_acquire )
		{
			// wait until the server signals that the specified amount of data is available.
			DWORD dwWaitResult = WaitForSingleObject(glb_event_handle, SYSTEM_EVENT_TIMEOUT);
			if (dwWaitResult != WAIT_OBJECT_0)
				throw std::logic_error("Error: The data ready event hasn't been triggered within a reasonable time.");

			size_t scans_available = 0;
			ret = GDS_GetData( connectionHandle, &scans_available, data_buffer, buffer_size_in_samples );
			if ( ret.ErrorCode )
			{
				std::cerr << "ERROR on GDS_GetData: " << ret.ErrorMessage << std::endl;
				break;
			}

			if ( scans_available > 0 )
			{
				total_acquired_scans += scans_available;
				file.write( (const char*) data_buffer, scans_available * buffer_size_per_scan * sizeof(float) );
				std::cout << CONSOLE_ESCAPE_CODE_CLEAR_TO_THE_LEFT << CONSOLE_ESCAPE_CODE_CARRIAGE_RETURN << std::flush;
				std::cout << total_acquired_scans << " from " << total_scans_to_acquire << " scans acquired";
			}
		}
		std::cout << std::endl;

		// Free the buffer and close the file.
		//-------------------------------------------------------------------------------------
		delete[] data_buffer;
		data_buffer = NULL;
		file.close();
	}
	catch(const std::logic_error & exc)
	{
		std::cerr << exc.what() << std::endl;
	}

	// Stop streaming data via the network.
	//-------------------------------------------------------------------------------------
	ret = GDS_StopStreaming(connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StopStreaming: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Command the device to stop acquiring data.
	//-------------------------------------------------------------------------------------
	ret = GDS_StopAcquisition(connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_StopAcquisition: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// Disconnect from the server.
	//-------------------------------------------------------------------------------------
	ret = GDS_Disconnect(&connectionHandle);
	if ( ret.ErrorCode )
	{
		std::cerr << "ERROR on GDS_Disconnect: " << ret.ErrorMessage << std::endl;
		std::cin.get();
		return -1;
	}

	// If the global handles exist then close them.
	//-------------------------------------------------------------------------------------
	if (glb_event_handle)
	{
		CloseHandle(glb_event_handle);
		glb_event_handle = NULL;
	}

	// Uninitialize the library.
	//-------------------------------------------------------------------------------------
	GDS_Uninitialize();

	std::cout << std::endl << "Press <return> to exit" << std::endl;
	std::cin.get();
	return 0;
}


// The method can be used to setup a simple config.
//-------------------------------------------------------------------------------------
void setup_config( GDS_GUSBAMP_CONFIGURATION* config, unsigned int sampling_rate )
{
	config->SampleRate = sampling_rate;
	config->NumberOfScans = 0;				//determine number of scans automatically
    config->ShortCutEnabled = FALSE;
	config->TriggerEnabled = FALSE;
	config->CounterEnabled = TRUE;
	config->InternalSignalGenerator.Enabled = FALSE;
	config->InternalSignalGenerator.Amplitude = 100;
	config->InternalSignalGenerator.Frequency = 10;
	config->InternalSignalGenerator.Offset = 0;
	config->InternalSignalGenerator.WaveShape = GDS_GUSBAMP_WAVESHAPE_SINE;

	for ( unsigned int i = 0; i < GDS_GUSBAMP_GROUPS_MAX; i++ )
	{
		config->CommonGround[i] = FALSE;
		config->CommonReference[i] = FALSE;
	}

	for ( unsigned int i = 0; i < GDS_GUSBAMP_CHANNELS_MAX; i++ )
	{
		config->Channels[i].Acquire = TRUE;
		config->Channels[i].BandpassFilterIndex = -1;
		config->Channels[i].NotchFilterIndex = -1;
		config->Channels[i].BipolarChannel = 0;
	}
}

// The method can be used to write the config to a console.
//-------------------------------------------------------------------------------------
void show_config( GDS_GUSBAMP_CONFIGURATION* config )
{
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "Config.SampleRate = " << config->SampleRate << std::endl;
	std::cout << "Config.NumberOfScans = " << config->NumberOfScans << std::endl;
	std::cout << "Config.ShortCutEnabled = " << config->ShortCutEnabled << std::endl;
	std::cout << "Config.TriggerEnabled = " << config->TriggerEnabled << std::endl;
	std::cout << "Config.CounterEnabled = " << config->CounterEnabled << std::endl;
	std::cout << "Config.InternalSignalGenerator.Enabled = " << config->InternalSignalGenerator.Enabled << std::endl;
	std::cout << "Config.InternalSignalGenerator.WaveShape = " << config->InternalSignalGenerator.WaveShape << std::endl;
	std::cout << "Config.InternalSignalGenerator.Frequency = " << config->InternalSignalGenerator.Frequency << std::endl;
	std::cout << "Config.InternalSignalGenerator.Amplitude = " << config->InternalSignalGenerator.Amplitude << std::endl;	
	std::cout << "Config.InternalSignalGenerator.Offset = " << config->InternalSignalGenerator.Offset << std::endl << std::endl;
	
	std::cout << "Config.CommonGround = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_GROUPS_MAX; i++ )
		std::cout << config->CommonGround[i] << "  "; 
	std::cout << std::endl;

	std::cout << "Config.CommonReference = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_GROUPS_MAX; i++ )
		std::cout << config->CommonReference[i] << "  "; 
	std::cout << std::endl;

	std::cout << "Config.Channels[i].Acquire = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_CHANNELS_MAX; i++ )
		std::cout << config->Channels[i].Acquire << "  "; 
	std::cout << std::endl;

	std::cout << "Config.Channels[i].BandpassFilterIndex = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_CHANNELS_MAX; i++ )
		std::cout << config->Channels[i].BandpassFilterIndex << "  "; 
	std::cout << std::endl;

	std::cout << "Config.Channels[i].NotchFilterIndex = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_CHANNELS_MAX; i++ )
		std::cout << config->Channels[i].NotchFilterIndex << "  "; 
	std::cout << std::endl;

	std::cout << "Config.Channels[i].BipolarChannel = ";
	for ( unsigned int i = 0; i < GDS_GUSBAMP_CHANNELS_MAX; i++ )
		std::cout << config->Channels[i].BipolarChannel << "  "; 
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
}

// The method is triggered periodically by the server as long as the specified
// amount of data is available.
//-------------------------------------------------------------------------------------
void on_data_ready_event(GDS_HANDLE connectionHandle, void* usrData)
{
	SetEvent(glb_event_handle);
}

// This method will be involved if the data acquisition struggles.
//-------------------------------------------------------------------------------------
void on_data_acquisition_error(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData)
{
	std::string* device = (std::string*) usrData;
	std::clog << "------------------------------------" << std::endl;
	std::clog << "Handle			= " << connectionHandle << std::endl;
	std::clog << "Device			= " << *device << std::endl;
	std::clog << "Where			= onDataAcquisitionError" << std::endl;
	std::clog << "What			= " << result.ErrorMessage << std::endl;
	std::clog << "ErrorCode		= " << result.ErrorCode << std::endl;
	std::clog << std::endl;
}

// This method will be involved if the server dies.
//-------------------------------------------------------------------------------------
void on_server_died_event(GDS_HANDLE connectionHandle, void* usrData)
{
	std::clog << "------------------------------------" << std::endl;
	std::clog << "Handle = " << connectionHandle << std::endl;
	std::clog << "What   = onServerDiedEvent" << std::endl; 
}