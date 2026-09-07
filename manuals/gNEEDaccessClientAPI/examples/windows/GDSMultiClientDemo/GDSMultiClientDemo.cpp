// GDSMultiClientDemo.cpp : Defines the entry point for the console application.
// COPYRIGHT © 2016 G.TEC MEDICAL ENGINEERING GMBH, AUSTRIA

#include "stdafx.h"
#include "DaqUnit.h"

#include <GDSClientAPI_gHIamp.h>

#include <string>
#include <vector>
#include <ctime>
#include <memory>

using namespace std;

vector<GDS_CONFIGURATION_BASE> MakeGHiAmpConfigurations(vector<string> deviceSerials, unsigned int samplingRate);
void FreeConfigurations(vector<GDS_CONFIGURATION_BASE> configurations);
std::string Now(const char* format = "%X");


int _tmain(int argc, _TCHAR* argv[])
{
	const unsigned int secondsToAcquire = 60;

	const char *deviceSerialsArray[] = { "HA-2021.05.08", "HA-2008.06.37" };
	vector<string> deviceSerials(deviceSerialsArray, deviceSerialsArray + sizeof(deviceSerialsArray) / sizeof(deviceSerialsArray[0]));

	//the server's endpoint
	GDS_ENDPOINT destination;
	strncpy(destination.IpAddress, "127.0.0.1", IP_ADDRESS_LENGTH_MAX);
	destination.Port = 50223;

	unsigned int samplingRate = 256;
	vector<GDS_CONFIGURATION_BASE> deviceConfigurations = MakeGHiAmpConfigurations(deviceSerials, samplingRate);

	//initialize GDS
	GDS_Initialize();

	try
	{
		vector<shared_ptr<DaqUnit>> daqUnits;

		for (size_t i = 0; i < deviceConfigurations.size(); i++)
		{
			cout << "Opening '" << deviceConfigurations[i].DeviceInfo.Name << "'..." << std::endl;

			//the client's endpoint
			GDS_ENDPOINT source;
			strncpy(source.IpAddress, "127.0.0.1", IP_ADDRESS_LENGTH_MAX);
			source.Port = 50224 + i;

			//create DAQ unit
			shared_ptr<DaqUnit> daqUnit = make_shared<DaqUnit>(vector<GDS_CONFIGURATION_BASE>(1, deviceConfigurations[i]), samplingRate, destination, source);
			daqUnits.push_back(daqUnit);

			//start data acquisition
			daqUnit->Start("data_" + std::to_string(i) + ".bin");
		}

		cout << std::endl << "Acquiring data for " << std::to_string((unsigned long long) secondsToAcquire) << " seconds... (" << Now() << ")" << std::endl;

		//run data acquisition for some time
		Sleep(secondsToAcquire * 1000);

		cout << "Stopping data acquisition... ";

		//stop data acquisition
		for (size_t i = 0; i < daqUnits.size(); i++)
			daqUnits[i]->Stop();

		cout << "(" << Now() << ")" << std::endl;
	}
	catch (GDSException &ex)
	{
		cout << "  ERROR: " << ex.ErrorMessage() << " (#" << ex.ErrorCode() << ")" << std::endl;
	}
	catch (std::exception &ex)
	{
		cout << "  ERROR: " << ex.what() << std::endl;
	}

	//release allocated resources
	FreeConfigurations(deviceConfigurations);

	//uninitialize GDS
	GDS_Uninitialize();

	cout << std::endl << std::endl << "Press any key to shutdown . . .";
	getchar();

	return 0;
}

vector<GDS_CONFIGURATION_BASE> MakeGHiAmpConfigurations(vector<string> deviceSerials, unsigned int samplingRate)
{
	vector<GDS_CONFIGURATION_BASE> configurations;

	for (size_t i = 0; i < deviceSerials.size(); i++)
	{
		GDS_CONFIGURATION_BASE configuration;
		
		//fill device info
		configuration.DeviceInfo.DeviceType = ::GDS_DEVICE_TYPE_GHIAMP;
		strncpy(configuration.DeviceInfo.Name, deviceSerials[i].c_str(), DEVICE_NAME_LENGTH_MAX);

		//create device configuration
		GDS_GHIAMP_CONFIGURATION *deviceConfig = new GDS_GHIAMP_CONFIGURATION();

		deviceConfig->SamplingRate = samplingRate;
		deviceConfig->NumberOfScans = 0;
		deviceConfig->CounterEnabled = TRUE;
		deviceConfig->TriggerLinesEnabled = FALSE;
		deviceConfig->HoldEnabled = FALSE;
		deviceConfig->InternalSignalGenerator.Enabled = TRUE;
		deviceConfig->InternalSignalGenerator.Frequency = 10;
		
		for (int i = 0; i < GDS_GHIAMP_CHANNELS_MAX; i++)
		{
			deviceConfig->Channels[i].Acquire = i < 40;
			deviceConfig->Channels[i].BandpassFilterIndex = -1;
			deviceConfig->Channels[i].NotchFilterIndex = -1;
			deviceConfig->Channels[i].ReferenceChannel = 0;
		}
		
		//add configuration to list
		configuration.Configuration = deviceConfig;
		configurations.push_back(configuration);
	}

	return configurations;
}

void FreeConfigurations(vector<GDS_CONFIGURATION_BASE> configurations)
{
	for (size_t i = 0; i < configurations.size(); i++)
		delete configurations[i].Configuration;
}

std::string Now(const char* format)
{
    std::time_t t = std::time(0);
    char cstr[128];
    std::strftime(cstr, sizeof(cstr), format, std::localtime(&t));
    return cstr;
}
