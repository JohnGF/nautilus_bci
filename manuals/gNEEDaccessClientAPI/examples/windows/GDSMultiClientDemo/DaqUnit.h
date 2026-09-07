// COPYRIGHT © 2016 G.TEC MEDICAL ENGINEERING GMBH, AUSTRIA
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cerrno>
#include <windows.h>
#include <process.h>
#include <GDSClientAPI.h>

#include "GDSException.h"
#include "InvalidOperationException.h"

using namespace std;


class DaqUnit
{
public:
	DaqUnit(vector<GDS_CONFIGURATION_BASE> devices, unsigned int samplingRate, GDS_ENDPOINT destination, GDS_ENDPOINT source) throw(GDSException);
	virtual ~DaqUnit(void);

	//! Starts data acquisition and writes received data into the specified file.
	/*!
		\param filename	The path and name of the file to create and to store data to, or empty to not store received data.

		\exception GDSException					Data acquisition cannot be started on the device.
		\exception InvalidOperationException	Data acquisition is already running.
		\exception inalid_argument				The file with the specified filename could not be created.
	 */
	void Start(string filename) throw(GDSException, InvalidOperationException, invalid_argument);
	void Stop() throw(GDSException);

protected:
	void HandleError(GDS_RESULT result);
	string GetIdentifier();

private:
	static void DoAcquisition(void *pParam);
	static void __stdcall GDS_DataReady(GDS_HANDLE connectionHandle, void *usrData);
	static void __stdcall GDS_DataAcquisitionError(GDS_HANDLE connectionHandle, GDS_RESULT result, void* usrData);

private:
	GDS_HANDLE _connectionHandle;
	BOOL _isCreator;
	vector<GDS_CONFIGURATION_BASE> _devices;
	unsigned int _samplingRate;
	HANDLE _dataAcquisitionThread;
	volatile bool _isRunning;
	HANDLE _dataReadyEvent;
	FILE *_file;
	SYSTEMTIME _time;
};

