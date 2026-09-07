# g.NEEDaccess Client API — Python vs C Comparison

The Python API (`pygds`) is a CFFI-based wrapper that loads the same native C DLL at runtime. While the Python API covers the core workflow, **the C API exposes significantly more functionality**.

---

## Summary

| | C API | Python API |
|---|---|---|
| **Total functions** | 69 | ~34 public methods + helpers |
| **Callbacks** | 9 registration functions | Not exposed |
| **Session management** | `BecomeCreator` / `BecomeParticipant` | Not exposed |
| **Async impedance** | `Start/StopImpedanceMeasurement(Ex)` | Not exposed (sync only) |
| **Multi-device abstraction** | Manual per-device calls | Single `GDS` object handles all |
| **Threaded data acquisition** | Manual (`GDS_GetData` polling) | Built-in double-buffered thread |
| **Live visualization** | Not included | `Scope` class (matplotlib) |

---

## Source Files

| Layer | Location |
|-------|----------|
| C headers | `C/GDSClientAPI.h`, `GDSClientAPI_gNautilus.h`, `GDSClientAPI_gUSBamp.h`, `GDSClientAPI_gHIamp.h` |
| Python wrapper | `Python/pygds-1.24.0-py3-none-any.whl` |
| Documentation | `documentation/` (PDFs for C, .NET, Python, Network) |
| Python examples | `examples/windows/GDSClientAPIDemoPython/` |

---

## C API Functions (69 total)

### Core API (`GDSClientAPI.h`)

| Function | Purpose |
|----------|---------|
| `GDS_Initialize()` | Initialize the GDS library |
| `GDS_Uninitialize()` | Uninitialize the GDS library |
| `GDS_GetConnectedDevices()` | List devices connected to a GDS server |
| `GDS_FreeConnectedDevicesList()` | Free memory from GetConnectedDevices |
| `GDS_Connect()` | Connect to server and create/open a session |
| `GDS_Disconnect()` | Dissociate handle and close connection |
| `GDS_BecomeCreator()` | Elevate participant to session creator |
| `GDS_BecomeParticipant()` | Downgrade creator to participant |
| `GDS_SetConfiguration()` | Set device configuration |
| `GDS_GetConfiguration()` | Get current device configuration |
| `GDS_FreeConfigurationList()` | Free memory from GetConfiguration |
| `GDS_StartAcquisition()` | Start data acquisition (creator only) |
| `GDS_StopAcquisition()` | Stop data acquisition |
| `GDS_StartStreaming()` | Enable data streaming for a handle |
| `GDS_StopStreaming()` | Disable data streaming for a handle |
| `GDS_GetDataInfo()` | Get data organization info (channels, buffer sizes) |
| `GDS_GetData()` | Read scans into a buffer |

### Core Callbacks (`GDSClientAPI.h`)

| Function | Purpose |
|----------|---------|
| `GDS_SetConfigurationChangedCallback()` | Callback when configuration changes |
| `GDS_SetDataAcquisitionStartedCallback()` | Callback when acquisition starts |
| `GDS_SetDataAcquisitionStoppedCallback()` | Callback when acquisition stops |
| `GDS_SetDataAcquisitionErrorCallback()` | Callback on acquisition error |
| `GDS_SetDataReadyCallback()` | Callback when data scans are available |
| `GDS_SetSessionCreatorDiedCallback()` | Callback when session creator dies |
| `GDS_SetNewSessionCreatorCallback()` | Callback when new creator is elected |
| `GDS_SetServerDiedCallback()` | Callback when server shuts down |
| `GDS_SetForcedClientShutdownCallback()` | Callback on forced client shutdown |

### g.Nautilus-specific (`GDSClientAPI_gNautilus.h`)

| Function | Purpose |
|----------|---------|
| `GDS_GNAUTILUS_GetDeviceInformation()` | Get extended device info string |
| `GDS_GNAUTILUS_GetChannelNames()` | Get electrode names and module count |
| `GDS_GNAUTILUS_GetAvailableChannels()` | Get boolean flags for available channels |
| `GDS_GNAUTILUS_GetAvailableDigitalIOs()` | Get digital I/O channel info |
| `GDS_GNAUTILUS_GetSupportedSamplingRates()` | List supported sampling rates |
| `GDS_GNAUTILUS_GetSupportedSensitivities()` | List supported sensitivity values |
| `GDS_GNAUTILUS_GetSupportedNetworkChannels()` | List supported wireless channels |
| `GDS_GNAUTILUS_GetSupportedInputSources()` | List supported input signal types |
| `GDS_GNAUTILUS_SetNetworkChannel()` | Set wireless radio channel |
| `GDS_GNAUTILUS_GetNetworkChannel()` | Get current wireless channel |
| `GDS_GNAUTILUS_GetScaling()` | Get scaling values |
| `GDS_GNAUTILUS_SetScaling()` | Set scaling values |
| `GDS_GNAUTILUS_ResetScaling()` | Reset scaling to neutral (0/1) |
| `GDS_GNAUTILUS_Calibrate()` | Calculate new scaling values |
| `GDS_GNAUTILUS_GetBandpassFilters()` | List available bandpass filters |
| `GDS_GNAUTILUS_GetNotchFilters()` | List available notch filters |
| `GDS_GNAUTILUS_GetImpedance()` | Measure impedance (deprecated) |
| `GDS_GNAUTILUS_GetImpedanceEx()` | Measure impedance (extended) |
| `GDS_GNAUTILUS_StartImpedanceMeasurement()` | Start async impedance measurement (deprecated) |
| `GDS_GNAUTILUS_StartImpedanceMeasurementEx()` | Start async impedance measurement (extended) |
| `GDS_GNAUTILUS_StopImpedanceMeasurement()` | Stop async impedance measurement |

### g.USBamp-specific (`GDSClientAPI_gUSBamp.h`)

| Function | Purpose |
|----------|---------|
| `GDS_GUSBAMP_GetDeviceInformation()` | Get extended device info string |
| `GDS_GUSBAMP_GetSupportedSamplingRates()` | List supported sampling rates with features |
| `GDS_GUSBAMP_GetBandpassFilters()` | List available bandpass filters |
| `GDS_GUSBAMP_GetNotchFilters()` | List available notch filters |
| `GDS_GUSBAMP_GetAsyncDigitalIOs()` | Get async digital I/O channel values |
| `GDS_GUSBAMP_SetAsyncDigitalOutputs()` | Set async digital output values |
| `GDS_GUSBAMP_GetScaling()` | Get scaling values |
| `GDS_GUSBAMP_SetScaling()` | Set scaling values |
| `GDS_GUSBAMP_Calibrate()` | Calculate new scaling values |
| `GDS_GUSBAMP_GetImpedance()` | Measure impedance (deprecated) |
| `GDS_GUSBAMP_GetImpedanceEx()` | Measure impedance (extended) |

### g.HIamp-specific (`GDSClientAPI_gHIamp.h`)

| Function | Purpose |
|----------|---------|
| `GDS_GHIAMP_GetDeviceInformation()` | Get extended device info string |
| `GDS_GHIAMP_GetAvailableChannels()` | Get boolean flags for available channels |
| `GDS_GHIAMP_GetSupportedSamplingRates()` | List supported sampling rates with features |
| `GDS_GHIAMP_GetBandpassFilters()` | List available bandpass filters |
| `GDS_GHIAMP_GetNotchFilters()` | List available notch filters |
| `GDS_GHIAMP_GetFactoryScaling()` | Get factory scaling values |
| `GDS_GHIAMP_GetScaling()` | Get scaling values |
| `GDS_GHIAMP_SetScaling()` | Set scaling values |
| `GDS_GHIAMP_Calibrate()` | Calculate new scaling values |
| `GDS_GHIAMP_GetImpedance()` | Measure impedance (deprecated) |
| `GDS_GHIAMP_GetImpedanceEx()` | Measure impedance (extended) |

---

## Python API (`pygds.GDS` class)

### Global Functions

| Function | Wraps |
|----------|-------|
| `Initialize()` | `GDS_Initialize()` |
| `Uninitialize()` | `GDS_Uninitialize()` |

### `GDS` Class Methods

| Method | Wraps C Function(s) |
|--------|---------------------|
| `GDS.__init__()` | `GDS_Connect()` |
| `GDS.SetConfiguration()` | `GDS_SetConfiguration()` |
| `GDS.GetConfiguration()` | `GDS_GetConfiguration()` |
| `GDS.GetDataInfo()` | `GDS_GetDataInfo()` |
| `GDS.GetData()` | `GDS_StartAcquisition()`, `GDS_StartStreaming()`, `GDS_GetData()`, `GDS_StopStreaming()`, `GDS_StopAcquisition()` |
| `GDS.GetAvailableChannels()` | `GDS_GNAUTILUS_GetAvailableChannels()`, `GDS_GHIAMP_GetAvailableChannels()` |
| `GDS.GetAvailableDigitalIOs()` | `GDS_GNAUTILUS_GetAvailableDigitalIOs()` |
| `GDS.GetAsyncDigitalIOs()` | `GDS_GUSBAMP_GetAsyncDigitalIOs()` |
| `GDS.SetAsyncDigitalOutputs()` | `GDS_GUSBAMP_SetAsyncDigitalOutputs()` |
| `GDS.GetDeviceInformation()` | Device-specific `GetDeviceInformation()` |
| `GDS.GetImpedance()` | Device-specific `GetImpedance()` |
| `GDS.GetImpedanceEx()` | Device-specific `GetImpedanceEx()` |
| `GDS.GetScaling()` | Device-specific `GetScaling()` |
| `GDS.Calibrate()` | Device-specific `Calibrate()` |
| `GDS.SetScaling()` | Device-specific `SetScaling()` |
| `GDS.ResetScaling()` | `GDS_GNAUTILUS_ResetScaling()` |
| `GDS.GetNetworkChannel()` | `GDS_GNAUTILUS_GetNetworkChannel()` |
| `GDS.GetFactoryScaling()` | `GDS_GHIAMP_GetFactoryScaling()` |
| `GDS.GetSupportedSamplingRates()` | Device-specific `GetSupportedSamplingRates()` |
| `GDS.GetBandpassFilters()` | Device-specific `GetBandpassFilters()` |
| `GDS.GetNotchFilters()` | Device-specific `GetNotchFilters()` |
| `GDS.GetSupportedSensitivities()` | `GDS_GNAUTILUS_GetSupportedSensitivities()` |
| `GDS.GetSupportedNetworkChannels()` | `GDS_GNAUTILUS_GetSupportedNetworkChannels()` |
| `GDS.GetSupportedInputSources()` | `GDS_GNAUTILUS_GetSupportedInputSources()` |
| `GDS.GetChannelNames()` | `GDS_GNAUTILUS_GetChannelNames()` |
| `GDS.SetNetworkChannel()` | `GDS_GNAUTILUS_SetNetworkChannel()` |
| `GDS.Close()` | `GDS_Disconnect()`, `GDS_FreeConfigurationList()` |

### Python-only Helpers

| Helper | Purpose |
|--------|---------|
| `Scope` class | Live matplotlib oscilloscope |
| `N_ch_calc()` | Channel count calculation |
| `NumberOfScans_calc()` | Auto-calculate recommended scans |
| `IndexAfter()` | Channel index navigation |
| `configure_demo()` | Pre-built demo configuration |
| `demo_*()` functions | Demo scripts (counter, save, scope, impedance, etc.) |

---

## Features Missing from Python

### 1. Callback Registration (all 9 functions)

The C API provides event-driven programming via callbacks. The Python API uses polling/threading instead.

```c
// C: Register a callback for data-ready events
GDS_SetDataReadyCallback(handle, my_callback, user_data);
```

### 2. Session Creator Management

```c
// C: Transfer session creator role
GDS_BecomeCreator(handle);
GDS_BecomeParticipant(handle);
```

### 3. Asynchronous Impedance Measurement (g.Nautilus)

```c
// C: Async impedance measurement
GDS_GNAUTILUS_StartImpedanceMeasurementEx(handle, ...);
GDS_GNAUTILUS_StopImpedanceMeasurement(handle);
```

Python only supports synchronous `GetImpedance()` / `GetImpedanceEx()`.

---

## Accessing C-only Features from Python

The Python wrapper loads all C symbols via CFFI. You can call raw C functions directly:

```python
from pygds import Initialize, _ffi_dll

Initialize()

# Call a C-only function directly via FFI
# (e.g., to register a callback)
```

> **Note:** This is undocumented and requires knowledge of the C API types/structs.

---

## What Python Adds Over C

- **Multi-device abstraction**: Single `GDS` object transparently manages multiple devices
- **Threaded double-buffered data acquisition**: `GetData()` runs a background thread
- **Unified configuration field names**: Maps device-specific names (e.g., `TriggerEnabled` / `TriggerLinesEnabled` / `DigitalIOs`) to common names
- **Live visualization**: `Scope` class for real-time plotting
- **Automatic cleanup**: `atexit` registration ensures `Uninitialize()` is called
- **Remote connection auto-detection**: `_this_ip()` auto-determines local IP
