//COPYRIGHT © 2016 G.TEC MEDICAL ENGINEERING GMBH, AUSTRIA
#include "StdAfx.h"
#include "DaqUnit.h"


DaqUnit::DaqUnit(vector<GDS_CONFIGURATION_BASE> devices, unsigned int samplingRate, GDS_ENDPOINT destination, GDS_ENDPOINT source)
	: _connectionHandle(NULL),
	_isCreator(FALSE),
	_devices(devices),
	_samplingRate(samplingRate),
	_dataAcquisitionThread(NULL),
	_isRunning(false),
	_dataReadyEvent(NULL),
	_file(NULL)
{
	//translate C++ vector to ansi C array
	char (*serialNumbersArray)[32] = new char[devices.size()][32];
	GDS_CONFIGURATION_BASE *configurationsArray = new GDS_CONFIGURATION_BASE[devices.size()];

	for (size_t i = 0; i < devices.size(); i++)
	{
		strncpy(serialNumbersArray[i], devices[i].DeviceInfo.Name, DEVICE_NAME_LENGTH_MAX);
		configurationsArray[i] = devices[i];
	}

	//connect to GDS and establish a DAQ session with the specified devices
	HandleError(GDS_Connect(destination, source, serialNumbersArray, devices.size(), false, &_connectionHandle, &_isCreator));

	//configure devices for acquisition
	HandleError(GDS_SetConfiguration(_connectionHandle, configurationsArray, devices.size()));
	delete [] configurationsArray;

	//set callbacks
	double dataReadyEventSeconds = 0.03;

	GDS_SetDataReadyCallback(_connectionHandle, &GDS_DataReady, (size_t) ceil(samplingRate * dataReadyEventSeconds), this);
	GDS_SetDataAcquisitionErrorCallback(_connectionHandle, &GDS_DataAcquisitionError, this);

	//initialize events
	_dataReadyEvent = CreateEvent(NULL, false, false, NULL);
}

DaqUnit::~DaqUnit(void)
{
	try
	{
		//stop data acquisition if running
		Stop();
	}
	catch (GDSException &ex)
	{
		if (ex.ErrorCode() != GDS_ERROR_INVALID_ACQUISITION_STATE)
			cout << "  ERROR (" << GetIdentifier() << "): Couldn't stop data acquisition unit: " << ex.ErrorMessage() << " (#" << ex.ErrorCode() << ")" << std::endl;
	}

	try
	{
		//close DAQ session and disconncet from GDS
		HandleError(GDS_Disconnect(&_connectionHandle));
	}
	catch (GDSException &ex)
	{
		cout << "  ERROR (" << GetIdentifier() << "): Couldn't close data acquisition unit: " << ex.ErrorMessage() << " (#" << ex.ErrorCode() << ")" << std::endl;
	}

	//release allocated resources
	CloseHandle(_dataReadyEvent);
}

void DaqUnit::Start(string filename)
{
	cout << "Starting " << _connectionHandle << " (" << GetIdentifier() << ")" << "..." << std::endl;

	if (_dataAcquisitionThread != NULL)
		throw InvalidOperationException("Data acquisition is already running.");

	ResetEvent(_dataReadyEvent);

	//start acquisition
	if (_isCreator)
	{
		HandleError(GDS_StartAcquisition(_connectionHandle));
		cout << "  acquisition started " << _connectionHandle << std::endl;
	}

	HandleError(GDS_StartStreaming(_connectionHandle));
	cout << "  streaming started " << _connectionHandle << std::endl;

	//open file
	if (filename.size() > 0)
	{
		_file = fopen(filename.c_str(), "wb");

		if (_file == NULL)
			throw invalid_argument("Error opening file '" + filename + "': " + strerror(errno) + " (system error code #" + std::to_string((long long) errno) + ").");
	}

	//start data acquisition thread
	_isRunning = true;
	_dataAcquisitionThread = (HANDLE) _beginthread(DoAcquisition, 0, this);
}

void DaqUnit::Stop()
{
	cout << "Stopping " << _connectionHandle << " (" << GetIdentifier() << ")" << "..." << std::endl;

	//stop data acquisition thread
	if (_dataAcquisitionThread != NULL)
	{
		_isRunning = false;
		SetEvent(_dataReadyEvent);
		WaitForSingleObject(_dataAcquisitionThread, INFINITE);
		_dataAcquisitionThread = NULL;
	}

	//stop acquisition
	HandleError(GDS_StopStreaming(_connectionHandle));
	cout << "  streaming stopped " << _connectionHandle << std::endl;

	if (_isCreator)
	{
		HandleError(GDS_StopAcquisition(_connectionHandle));
		cout << "  acquisition stopped " << _connectionHandle << std::endl;
	}
}

void DaqUnit::HandleError(GDS_RESULT result)
{
	if (result.ErrorCode != GDS_ERROR_SUCCESS)
		throw GDSException(result);
}

string DaqUnit::GetIdentifier()
{
	string daqUnitIdentifier;
		
	for (size_t i = 0; i < _devices.size(); i++)
		daqUnitIdentifier += (string((i == 0) ? "" : ", ") + string(_devices[i].DeviceInfo.Name));

	return daqUnitIdentifier;
}

void DaqUnit::DoAcquisition(void *pParam)
{
	DaqUnit *daqUnit = (DaqUnit*) pParam;
	float *buffer = NULL;

	try
	{
		size_t scanCount = 1;
		size_t channelsPerDeviceCount = 0;
		size_t scanSizeSamples = 0;

		daqUnit->HandleError(GDS_GetDataInfo(daqUnit->_connectionHandle, &scanCount, NULL, &channelsPerDeviceCount, &scanSizeSamples));

		//allocate buffers
		size_t bufferSizeSeconds = 1;
		size_t bufferSize = scanSizeSamples * daqUnit->_samplingRate * bufferSizeSeconds;
		buffer = new float[bufferSize];

		while(daqUnit->_isRunning)
		{
			scanCount = 0;

			//wait until new data is ready
			WaitForSingleObject(daqUnit->_dataReadyEvent, INFINITE);
			//cout << "getting data " << daqUnit->_connectionHandle << " (running == " << daqUnit->_isRunning << ")" << std::endl;

			//retrieve data
			daqUnit->HandleError(GDS_GetData(daqUnit->_connectionHandle, &scanCount, buffer, bufferSize));

			//GetSystemTime(&daqUnit->_time);
			//cout << daqUnit->_time.wHour << ":" << daqUnit->_time.wMinute << ":" << daqUnit->_time.wSecond << ":" << daqUnit->_time.wMilliseconds << " GDS received " << scanCount << " scans from handle " << daqUnit->_connectionHandle << " (thread id " << _threadid << ")" << std::endl;

			if (scanCount > 0 && daqUnit->_file != NULL)
				fwrite(buffer, sizeof(float), scanCount * scanSizeSamples, daqUnit->_file);

		}
	}
	catch (GDSException &ex)
	{
		cout << "  ERROR (" << daqUnit->GetIdentifier() << "): " << ex.ErrorMessage() << " (#" << ex.ErrorCode() << ")" << std::endl;
	}
	catch (std::exception &ex)
	{
		cout << "  ERROR (" << daqUnit->GetIdentifier() << "): " << ex.what() << std::endl;
	}

	//release allocated resources
	if (daqUnit->_file != NULL)
	{
		fclose(daqUnit->_file);
		daqUnit->_file = NULL;
	}

	ResetEvent(daqUnit->_dataReadyEvent);
	daqUnit->_isRunning = false;

	delete[] buffer;
}

void DaqUnit::GDS_DataReady(GDS_HANDLE connectionHandle, void *usrData)
{
	DaqUnit *daqUnit = (DaqUnit*) usrData;

	//GetSystemTime(&daqUnit->_time);
	//cout << daqUnit->_time.wHour <<":"<< daqUnit->_time.wMinute << ":" << daqUnit->_time.wSecond << ":" << daqUnit->_time.wMilliseconds << " GDS data ready event from handle " << connectionHandle << std::endl;
	
	if (connectionHandle != daqUnit->_connectionHandle)
		return;

	SetEvent(daqUnit->_dataReadyEvent);
}

void DaqUnit::GDS_DataAcquisitionError(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData)
{
	DaqUnit *daqUnit = (DaqUnit*) usrData;

	if (connectionHandle != daqUnit->_connectionHandle)
		return;

	cout << "  ERROR (" << daqUnit->GetIdentifier() << "): " << result.ErrorMessage << " (#" << result.ErrorCode << ")" << std::endl;
}