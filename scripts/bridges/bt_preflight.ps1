param(
    [string]$Address = "682719F8B937",
    [string]$NamePattern = "OctaMon|OptoMon|PortaLite|Artinis",
    [int]$ScanSeconds = 12,
    [switch]$NoPair
)
# Bluetooth preflight for Artinis devices. Emits one JSON object on stdout.
$ErrorActionPreference = "Stop"
$r = [ordered]@{ radio = "unknown"; radio_fixed = $false; found = $false; paired = $false;
                 connected = $false; address = $Address; name = $null; action = @(); ok = $false; error = $null }
try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Devices.Radios.Radio, Windows.System.Devices, ContentType = WindowsRuntime]
    $null = [Windows.Devices.Bluetooth.BluetoothDevice, Windows.Devices.Bluetooth, ContentType = WindowsRuntime]
    $null = [Windows.Devices.Enumeration.DeviceInformation, Windows.Devices.Enumeration, ContentType = WindowsRuntime]

    $asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    function Await($op, [Type]$type) {
        $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
        $null = $t.Wait(-1)
        $t.Result
    }

    # 1. Radio state
    $null = [Windows.Devices.Radios.RadioAccessStatus, Windows.System.Devices, ContentType = WindowsRuntime]
    $null = Await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus])
    $radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
    $bt = @($radios | Where-Object { $_.Kind -eq 'Bluetooth' })
    if ($bt.Count -eq 0) { $r.radio = "none"; throw "No Bluetooth radio available (check adapter / driver)" }
    foreach ($rad in $bt) {
        if ($rad.State -ne 'On') {
            $null = Await ($rad.SetStateAsync('On')) ([Windows.Devices.Radios.RadioAccessStatus])
            $r.radio_fixed = $true; $r.action += "radio_on"
        }
    }
    $r.radio = "on"

    # 2. Locate device (paired cache first)
    $addrInt = [Convert]::ToUInt64($Address, 16)
    $dev = Await ([Windows.Devices.Bluetooth.BluetoothDevice]::FromBluetoothAddressAsync($addrInt)) ([Windows.Devices.Bluetooth.BluetoothDevice])
    $pairedNow = $dev -ne $null -and $dev.DeviceInformation.Pairing.IsPaired
    if (-not $pairedNow) {
        # Fallback: PnP registry knows bonded classic devices even when WinRT returns null (SSH session)
        $pnp = Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue |
               Where-Object { $_.InstanceId -match "DEV_$Address" } | Select-Object -First 1
        if ($pnp) { $pairedNow = $true; $r.name = $pnp.FriendlyName; $r.action += "paired_via_pnp" }
    }

    # 3. Not paired: scan for it by address or name, then pair
    if (-not $pairedNow) {
        $r.action += "scan"
        $sel = [Windows.Devices.Bluetooth.BluetoothDevice]::GetDeviceSelectorFromPairingState($false)
        $props = @("System.Devices.Aep.DeviceAddress", "System.Devices.Aep.IsPaired")
        $kind = [Windows.Devices.Enumeration.DeviceInformationKind, Windows.Devices.Enumeration, ContentType = WindowsRuntime]::AssociationEndpoint
        $w = [Windows.Devices.Enumeration.DeviceInformation]::CreateWatcher([string]$sel, [string[]]$props, $kind)
        $script:hit = $null
        $addrNorm = ($Address -replace '[:\-]', '').ToUpper()
        $null = Register-ObjectEvent $w Added -MessageData @($addrNorm, $NamePattern) -Action {
            $di = $EventArgs; $a = $Event.MessageData[0]; $p = $Event.MessageData[1]
            $da = [string]$di.Properties["System.Devices.Aep.DeviceAddress"]
            if (($da -replace '[:\-]', '').ToUpper() -eq $a -or $di.Name -match $p) { $global:BtHit = $di }
        }
        $global:BtHit = $null
        $w.Start()
        $end = (Get-Date).AddSeconds($ScanSeconds)
        while ((Get-Date) -lt $end -and -not $global:BtHit) { Start-Sleep -Milliseconds 300 }
        $w.Stop()
        Get-EventSubscriber | Unregister-Event
        if ($global:BtHit) {
            $r.found = $true; $r.name = $global:BtHit.Name
            if (-not $NoPair) {
                $res = Await ($global:BtHit.Pairing.PairAsync()) ([Windows.Devices.Enumeration.DevicePairingResult])
                $r.action += "pair:" + $res.Status
                $dev = Await ([Windows.Devices.Bluetooth.BluetoothDevice]::FromBluetoothAddressAsync($addrInt)) ([Windows.Devices.Bluetooth.BluetoothDevice])
                $pairedNow = $dev -ne $null -and $dev.DeviceInformation.Pairing.IsPaired
            }
        } else {
            throw "Device $Address not found. Power on the OctaMon and keep it close to the PC."
        }
    } else {
        $r.found = $true; if ($dev) { $r.name = $dev.Name }
    }

    $r.paired = [bool]$pairedNow
    if (-not $pairedNow) { throw "Device found but not paired (it may need a PIN: pair once in Windows Settings)." }

    # 4. Wake / verify link by opening an uncached RFCOMM service query
    $null = [Windows.Devices.Bluetooth.BluetoothCacheMode, Windows.Devices.Bluetooth, ContentType = WindowsRuntime]
    $null = [Windows.Devices.Bluetooth.Rfcomm.RfcommDeviceServicesResult, Windows.Devices.Bluetooth, ContentType = WindowsRuntime]
    if ($dev) {
    $svc = Await ($dev.GetRfcommServicesAsync([Windows.Devices.Bluetooth.BluetoothCacheMode]::Uncached)) ([Windows.Devices.Bluetooth.Rfcomm.RfcommDeviceServicesResult])
    $r.action += "link_query:" + $svc.Error
    $r.connected = ($dev.ConnectionStatus -eq 'Connected') -or ($svc.Services.Count -gt 0)
    }
    $r.ok = $true
} catch {
    $r.error = $_.Exception.Message
}
$r | ConvertTo-Json -Compress
