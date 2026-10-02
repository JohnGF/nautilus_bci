using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Net;
using System.Threading;

using Gtec.Gds.Client.API.Wrapper;
using System.Runtime.InteropServices;
using System.IO;

namespace GdsClientAPIDemoDotNet
{
    class Program
    {
        static GdsClientApiLibraryWrapper.GdsResultCallback DataAcquisitionErrorCallback = OnDataAcquisitionError;
        static GdsClientApiLibraryWrapper.GdsCallback DataReadyCallback = OnDataReadyCallback;

        /// <summary>
        /// The amount of data to wait for before the <see cref="_dataReadyEvent"/> is triggered in milliseconds.
        /// </summary>
        const int DataReadyThresholdInMs = 30;

        /// <summary>
        /// The event that is signalled when new data is available.
        /// </summary>
        static AutoResetEvent _dataReadyEvent = new AutoResetEvent(false);
        
        /// <summary>
        /// The main entry point of the application.
        /// </summary>
        /// <param name="args">Command line arguments.</param>
        static void Main(string[] args)
        {
            //define the server and client endpoint addresses
            IPEndPoint remoteEndpoint = new IPEndPoint(IPAddress.Loopback, 50223);  //the address of the endpoint that runs the GDS server
            IPEndPoint localEndpoint = new IPEndPoint(IPAddress.Loopback, 50224);   //the address of the endpoint that runs the client/demo

            //initialize the library
            GdsClientApiLibraryWrapper.Initialize();

            try
            {
                //retrieve and print the list of devices that are connected to the server
                List<GdsClientApiLibraryWrapper.DeviceConnectionInfo> daqUnits = GdsClientApiLibraryWrapper.GetConnectedDevices(remoteEndpoint, localEndpoint);
                Dictionary<string, GdsClientApiLibraryWrapper.GdsDeviceType> serialNumberAndType = PrintConnectedDevices(daqUnits, remoteEndpoint);

                //let the user enter the serial number of the device to use
                Console.Write("Please enter the serial number of the device: ");
                string serialNumber = Console.ReadLine();

                //determine the device type and check whether the given serialNumber is valid or not
                GdsClientApiLibraryWrapper.GdsDeviceType deviceType;
                try
                {
                    deviceType = serialNumberAndType[serialNumber];
                }
                catch (KeyNotFoundException)
                {
                    Console.WriteLine("The given device serial number (" + serialNumber + ") is invalid.");
                    return;
                }

                //open the device
                bool isCreator;
                UInt64 connectionHandle = GdsClientApiLibraryWrapper.Connect(remoteEndpoint, localEndpoint, new string[] { serialNumber }, true, out isCreator);

                try
                {
                    uint bufferSizeInSeconds = 5;   //receive 5 seconds of data at maximum per query/block
                    uint sampleRate = 0;            //sample rate is set in 'CreateGUsbAmpConfiguration()'      
                    ulong[] channelsPerDevice;
                    ulong scanSizeInSamples;

                    //connect callback to get notified about errors in data acquisition
                    GdsClientApiLibraryWrapper.SetDataAcquisitionErrorCallback(connectionHandle, DataAcquisitionErrorCallback, IntPtr.Zero);

                    //configure devices
                    GUsbAmpGdsClientApiLibraryWrapper.GdsConfiguration[] deviceConfigurations = CreateConfiguration(serialNumber, deviceType, out sampleRate);
                    GdsClientApiLibraryWrapper.SetConfiguration(connectionHandle, deviceConfigurations);

                    //connect callback to trigger data processing each time when the specified amount of data is available
                    GdsClientApiLibraryWrapper.SetDataReadyCallback(connectionHandle, DataReadyCallback, (ulong) (DataReadyThresholdInMs / (double) 1000 * sampleRate), IntPtr.Zero);
                    
                    //determine size of a single scan
                    GdsClientApiLibraryWrapper.GetDataInfo(connectionHandle, 1, out channelsPerDevice, out scanSizeInSamples);

                    //create the (managed) buffer where a single block of received data should be written to and limit such a block to a specific amount of seconds and scans, respectively
                    ulong bufferSizeInSamples = bufferSizeInSeconds * sampleRate * scanSizeInSamples;
                    byte[] receiveBuffer = new byte[bufferSizeInSamples * sizeof(float)];           //(a float array of size 'bufferSizeInSamples' could be used here as well, but the BinaryWriter used later only supports writing arrays of type byte)

                    //allocate buffer in unmanaged memory and pin it so that the garbage collector doesn't move it
                    GCHandle receiveBufferHandle = GCHandle.Alloc(receiveBuffer, GCHandleType.Pinned);

                    try
                    {
                        //start acquisition on the device on the server
                        GdsClientApiLibraryWrapper.StartAcquisition(connectionHandle);

                        //start streaming in order to be able to receive data from the GDS server
                        GdsClientApiLibraryWrapper.StartStreaming(connectionHandle);

                        Console.WriteLine("\nData acquisition started at {0}\nPress ESC to stop...\n", DateTime.Now.ToString("HH:mm:ss"));

                        //create file where received data should be written to
                        using (BinaryWriter fileWriter = new BinaryWriter(File.Open("receivedData.bin", FileMode.Create)))
                        {
                            //acquire data as long as the escape key is pressed / main daq loop
                            while (!Console.KeyAvailable || Console.ReadKey(true).Key != ConsoleKey.Escape)
                            {
                                //wait until new data is provided
                                _dataReadyEvent.WaitOne();

                                //get as many scans as the buffer can hold or as are currently available
                                ulong scanCount = 0;
                                ulong receivedScans = GdsClientApiLibraryWrapper.GetData(connectionHandle, scanCount, receiveBufferHandle.AddrOfPinnedObject(), bufferSizeInSamples);

                                //if anything is received, write it to the file
                                if (receivedScans > 0)
                                    fileWriter.Write(receiveBuffer, 0, (int) (receivedScans * scanSizeInSamples * sizeof(float)));
                            }

                            Console.WriteLine("Data acquisition stopped at {0}", DateTime.Now.ToString("HH:mm:ss"));
                        }
                    }
                    finally
                    {
                        //stop acquisition and streaming
                        GdsClientApiLibraryWrapper.StopStreaming(connectionHandle);
                        GdsClientApiLibraryWrapper.StopAcquisition(connectionHandle);

                        //release allocated unmanaged resources
                        receiveBufferHandle.Free();
                    }
                }
                finally
                {
                    //close the connection to the server
                    GdsClientApiLibraryWrapper.Disconnect(ref connectionHandle);
                }
            }
            catch (GdsException ex)
            {
                //an error occurred while using the GDS Client API
                Console.WriteLine("ERROR #{0} ('{1}'): {2}", (int) ex.ErrorCode, ex.ErrorCode, ex.Message);
            }
            catch (DllNotFoundException)
            {
                //GDS device library is not found
                Console.WriteLine("The GDS device library is not found on the current path.");
            }
            catch (InvalidOperationException ex)
            {
                //no devices were found on the server
                Console.WriteLine(ex.Message);
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine("INVALID ARGUMENT ERROR: " + ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine("GENERAL ERROR:");
                Console.WriteLine(ex.StackTrace);
            }
            finally
            {
                //uninitialize the library
                GdsClientApiLibraryWrapper.Uninitialize();

                Console.WriteLine(Environment.NewLine);
                Console.WriteLine("Press ENTER to exit...");
                Console.ReadLine();
            }
        }

        /// <summary>
        /// Writes the list of devices that are currently connected to the server out to the console.
        /// </summary>
        /// <param name="daqUnits">The data structure which holds the device serial number as well as the device type.</param>
        /// <param name="remoteEndpoint">The endpoint of the GDS. This parameter is used only to print additional information on the console.</param>
        /// <remarks>Devices in use are grouped by the data acquisition unit they belong to.</remarks>
        /// <returns>A dictionary which maps the device serial number to its according device type.</returns>
        static Dictionary<string, GdsClientApiLibraryWrapper.GdsDeviceType> PrintConnectedDevices(List<GdsClientApiLibraryWrapper.DeviceConnectionInfo> daqUnits, IPEndPoint remoteEndpoint)
        {
            if (daqUnits.Count == 0)
                throw new InvalidOperationException(String.Format("No devices are connected to GDS server at endpoint '{0}'", remoteEndpoint.ToString()));

            //print device list
            Console.WriteLine("The following devices are connected to GDS server at endpoint '{0}':", remoteEndpoint.ToString());

            Dictionary<string, GdsClientApiLibraryWrapper.GdsDeviceType> serialNumberAndType = new Dictionary<string,GdsClientApiLibraryWrapper.GdsDeviceType>();
            foreach (GdsClientApiLibraryWrapper.DeviceConnectionInfo daqUnit in daqUnits)
            {
                Console.WriteLine();

                if (daqUnit.InUse)
                    Console.WriteLine("  Currently in use by a separate DAQ unit:");
                else
                    Console.WriteLine("  Connected, but currently not in use:");

                foreach (GdsClientApiLibraryWrapper.DeviceInfo connectedDevice in daqUnit.ConnectedDevices)
                {
                    serialNumberAndType.Add(connectedDevice.Name, connectedDevice.DeviceType);
                    Console.WriteLine("\t{0}\t({1})", connectedDevice.Name, connectedDevice.DeviceType.ToString());
                }
            }
            Console.WriteLine();
            return serialNumberAndType;
        }

        /// <summary>
        /// Initializes a default configuration for a g.USBamp device to use with GDS.
        /// </summary>
        /// <param name="serialNumber">The serial number of the g.USBamp device to use.</param>
        /// <param name="sampleRate">The sample rate of the device to use.</param>
        /// <returns>An initialized g.USBamp device configuration to use with GDS.</returns>
        static GUsbAmpGdsClientApiLibraryWrapper.GdsConfiguration CreateGUsbAmpConfiguration(string serialNumber, out uint sampleRate)
        {
            sampleRate = 256;
            const int NumberOfAcquiredChannels = 16;            //valid values range from 1...16

            //initialize device configuration structure
            GUsbAmpGdsClientApiLibraryWrapper.GdsGUsbAmpConfiguration deviceConfiguration = new GUsbAmpGdsClientApiLibraryWrapper.GdsGUsbAmpConfiguration();

            deviceConfiguration.SampleRate = sampleRate;
            deviceConfiguration.NumberOfScans = (UIntPtr) 0;    //use recommended number of scans by setting this value to zero
            deviceConfiguration.TriggerEnabled = true;
            deviceConfiguration.CounterEnabled = false;
            deviceConfiguration.ShortCutEnabled = false;
            deviceConfiguration.CommonGround = new bool[4] { false, false, false, false };
            deviceConfiguration.CommonReference = new bool[4] { false, false, false, false };

            deviceConfiguration.InternalSignalGenerator = new GUsbAmpGdsClientApiLibraryWrapper.GdsGUsbAmpSignalGeneratorConfiguration();
            deviceConfiguration.InternalSignalGenerator.Enabled = false;
            deviceConfiguration.InternalSignalGenerator.WaveShape = GUsbAmpGdsClientApiLibraryWrapper.GdsGUsbAmpWaveShape.Sine;
            deviceConfiguration.InternalSignalGenerator.Frequency = 10;
            deviceConfiguration.InternalSignalGenerator.Amplitude = 1;
            deviceConfiguration.InternalSignalGenerator.Offset = 0;

            deviceConfiguration.Channels = new GUsbAmpGdsClientApiLibraryWrapper.GdsGUsbAmpChannelConfiguration[GUsbAmpGdsClientApiLibraryWrapper.MaxNumberOfChannels];

            for (int i = 0; i < deviceConfiguration.Channels.Length; i++)
            {
                deviceConfiguration.Channels[i].Acquire = (i < NumberOfAcquiredChannels);
                deviceConfiguration.Channels[i].BandpassFilterIndex = -1;                   //the one-based index of the g.USBamp binary filter table (index 42 == bandpass 0.1-30 Hz, zero indicates 'no filter'
                deviceConfiguration.Channels[i].NotchFilterIndex = -1;                      //the one-based index of the g.USBamp binary filter table (index 2 == notch 50Hz), zero indicates 'no filter'
                deviceConfiguration.Channels[i].BipolarChannel = 0;
            }

            //build and return complete GDS configuration structure
            GdsClientApiLibraryWrapper.GdsConfiguration gdsConfig = new GdsClientApiLibraryWrapper.GdsConfiguration();

            gdsConfig.DeviceConfiguration = deviceConfiguration;
            gdsConfig.DeviceInfo = new GdsClientApiLibraryWrapper.DeviceInfo();
            gdsConfig.DeviceInfo.DeviceType = GdsClientApiLibraryWrapper.GdsDeviceType.GUsbAmp;
            gdsConfig.DeviceInfo.Name = serialNumber;

            return gdsConfig;
        }

        /// <summary>
        /// Initializes a default configuration for a g.USBamp device to use with GDS.
        /// </summary>
        /// <param name="serialNumber">The serial number of the g.HIamp device to use.</param>
        /// <param name="sampleRate">The sample rate of the device to use.</param>
        /// <returns>An initialized g.HIamp device configuration to use with GDS.</returns>
        static GHiAmpGdsClientApiLibraryWrapper.GdsConfiguration CreateGHiAmpConfiguration(string serialNumber, out uint sampleRate)
        {
            sampleRate = 256;
            const int numberOfAcquiredChannels = 40; //valid values range from 1...256

            GHiAmpGdsClientApiLibraryWrapper.GdsGHiAmpConfiguration deviceConfiguration = new GHiAmpGdsClientApiLibraryWrapper.GdsGHiAmpConfiguration();
            deviceConfiguration.SamplingRate = sampleRate;
            deviceConfiguration.NumberOfScans = 0;
            deviceConfiguration.HoldEnabled = false;
            deviceConfiguration.TriggerLinesEnabled = false;
            deviceConfiguration.CounterEnabled = true;
            deviceConfiguration.InternalSignalGenerator.Enabled = true;
            deviceConfiguration.InternalSignalGenerator.Frequency = 10;

            deviceConfiguration.Channels = new GHiAmpGdsClientApiLibraryWrapper.GdsGHiAmpChannelConfiguration[GHiAmpGdsClientApiLibraryWrapper.MaxNumberOfChannels];

            for (int i = 0; i < GHiAmpGdsClientApiLibraryWrapper.MaxNumberOfChannels; i++)
            {
                deviceConfiguration.Channels[i].Acquire = (i < numberOfAcquiredChannels);
                deviceConfiguration.Channels[i].BandpassFilterIndex = -1;
                deviceConfiguration.Channels[i].NotchFilterIndex = -1;
                deviceConfiguration.Channels[i].ReferenceChannel = 0;
            }

            GdsClientApiLibraryWrapper.GdsConfiguration gdsConfig = new GdsClientApiLibraryWrapper.GdsConfiguration();

            gdsConfig.DeviceConfiguration = deviceConfiguration;
            gdsConfig.DeviceInfo = new GdsClientApiLibraryWrapper.DeviceInfo();
            gdsConfig.DeviceInfo.DeviceType = GdsClientApiLibraryWrapper.GdsDeviceType.GHiAmp;
            gdsConfig.DeviceInfo.Name = serialNumber;
            
            return gdsConfig;
        }

        /// <summary>
        /// Initializes a default configuration for a g.Nautilus device to use with GDS.
        /// </summary>
        /// <param name="serialNumber">The serial number of the g.Nautilus device to use.</param>
        /// <param name="sampleRate">The sample rate of the device to use.</param>
        /// <returns>An initialized g.Nautilus device configuration to use with GDS.</returns>
        static GNautilusGdsClientApiLibraryWrapper.GdsConfiguration CreateGNautilusConfiguration(string serialNumber, out uint sampleRate)
        {
            sampleRate = 500;
            const int numberOfAcquiredChannels = 8; //valid values range from 1...8 (for a single module)
            const double sensitivity = 187500.0;

            GNautilusGdsClientApiLibraryWrapper.GdsGNautilusConfiguration deviceConfiguration = new GNautilusGdsClientApiLibraryWrapper.GdsGNautilusConfiguration();
            deviceConfiguration.IsSlave = false;
            deviceConfiguration.SamplingRate = sampleRate;
            deviceConfiguration.NumberOfScans = 0;
            deviceConfiguration.NetworkChannel = 0;
            deviceConfiguration.DigitalIOs = false;
            deviceConfiguration.InputSignal = GNautilusGdsClientApiLibraryWrapper.InputSignal.Electrode;
            deviceConfiguration.AccelerationData = false;
            deviceConfiguration.BatteryLevel = false;
            deviceConfiguration.CAR = false;
            deviceConfiguration.Counter = false;
            deviceConfiguration.LinkQualityInformation = false;
            deviceConfiguration.ValidationIndicator = false;
            deviceConfiguration.NoiseReduction = false;

            deviceConfiguration.Channels = new GNautilusGdsClientApiLibraryWrapper.GdsGNautilusChannelConfiguration[GNautilusGdsClientApiLibraryWrapper.MaxNumberOfChannels];

            for (int i = 0; i < GNautilusGdsClientApiLibraryWrapper.MaxNumberOfChannels; i++ )
            {
                deviceConfiguration.Channels[i].Enabled = (i < numberOfAcquiredChannels);
                deviceConfiguration.Channels[i].Sensitivity = sensitivity;
                deviceConfiguration.Channels[i].UsedForNoiseReduction = false;
                deviceConfiguration.Channels[i].BandpassFilterIndex = -1;
                deviceConfiguration.Channels[i].NotchFilterIndex = -1;
                deviceConfiguration.Channels[i].BipolarChannel = -1;
            }

            GdsClientApiLibraryWrapper.GdsConfiguration gdsConfig = new GdsClientApiLibraryWrapper.GdsConfiguration();

            gdsConfig.DeviceConfiguration = deviceConfiguration;
            gdsConfig.DeviceInfo = new GdsClientApiLibraryWrapper.DeviceInfo();
            gdsConfig.DeviceInfo.DeviceType = GdsClientApiLibraryWrapper.GdsDeviceType.GNautilus;
            gdsConfig.DeviceInfo.Name = serialNumber;

            return gdsConfig;
        }

        /// <summary>
        /// Initalizes a default device configuration according to the given device type.
        /// </summary>
        /// <param name="serialNumber">The serial number (i.e. the device name) of the device.</param>
        /// <param name="deviceType">The device type according to the serialnumber.</param>
        /// <param name="sampleRate">The sample rate of the device to use.</param>
        /// <returns>An initialized default configuration for the device.</returns>
        static GdsClientApiLibraryWrapper.GdsConfiguration[] CreateConfiguration(string serialNumber, GdsClientApiLibraryWrapper.GdsDeviceType deviceType, out uint sampleRate)
        {
            switch (deviceType)
            {
                case GdsClientApiLibraryWrapper.GdsDeviceType.GUsbAmp:
                    return new GUsbAmpGdsClientApiLibraryWrapper.GdsConfiguration[] { CreateGUsbAmpConfiguration(serialNumber, out sampleRate) };
                case GdsClientApiLibraryWrapper.GdsDeviceType.GHiAmp:
                    return new GUsbAmpGdsClientApiLibraryWrapper.GdsConfiguration[] { CreateGHiAmpConfiguration(serialNumber, out sampleRate) };
                case GdsClientApiLibraryWrapper.GdsDeviceType.GNautilus:
                    return new GNautilusGdsClientApiLibraryWrapper.GdsConfiguration[] { CreateGNautilusConfiguration(serialNumber, out sampleRate) };
                default:
                    throw new ArgumentException("The device " + serialNumber + " of type " + deviceType.ToString() + " is not supported yet.");
            }
        }

        /// <summary>
        /// Handles the GDS's <b>DataAcquisitionError</b> callback.
        /// Just prints information about the error that occurred. An application might react on the occurred error properly.
        /// </summary>
        /// <param name="connectionHandle">The connection handle for which the callback was triggered.</param>
        /// <param name="error">Details on the error that occurred.</param>
        /// <param name="userData">A pointer to an unmanaged block of memory containing the custom user data specified when setting the callback.</param>
        static void OnDataAcquisitionError(UInt64 connectionHandle, GdsClientApiLibraryWrapper.GdsResult error, IntPtr userData)
        {
            Console.WriteLine("DATA ACQUISITION ERROR #{0}: {1}", error.ErrorCode, error.ErrorMessage);
        }

        /// <summary>
        /// Handles the GDS's <b>DataReady</b> callback.
        /// Signals the application that the requested amount of data is available now.
        /// </summary>
        /// <param name="connectionHandle">The connection handle for which the callback was triggered.</param>
        /// <param name="userData">A pointer to an unmanaged block of memory containing the custom user data specified when setting the callback.</param>
        static void OnDataReadyCallback(UInt64 connectionHandle, IntPtr userData)
        {
            _dataReadyEvent.Set();
        }
    }
}
