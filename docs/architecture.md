# Architecture

The Nautilus BCI platform follows a modular **hub-and-spoke** model. The `nautilus_bci` system acts as the central hub, whereas individual device repositories and hardware bridges act as spokes.

A core principle is that `nautilus_bci` strictly consumes LSL streams and never imports device code directly.

## LSL is the Glue

The Lab Streaming Layer (LSL) provides the essential infrastructure linking the entire platform:
- **Shared Clock:** Time synchronization ensures streams originating from different devices or hosts align seamlessly.
- **Automatic Discovery:** Multicast and broadcast resolution naturally finds available streams.
- **Clock Synchronization:** Our recorders (`scripts/recorders/multimodal_bids_recorder.py` and `bids_recorder.py`) natively open inlets using `proc_clocksync`. This converts timestamps from incoming streams to the recording PC's local clock automatically.

**Important:** LSL discovery requires proper firewall configuration. Be sure to review the [LSL Network & Firewall Setup](guides/lsl_network_setup.md).

## Device Sources

Below is a table of the primary data sources, verified against the current codebase:

| Device / Source | Stream Name | Stream Type | Channels | Rate | Source script/repo |
|---|---|---|---|---|---|
| g.Nautilus | `gNautilus_EEG` or `gNautilus_fNIRS` | `EEG` or `fNIRS` | Varies (e.g. 32) | Configurable | `scripts/bridges/gds_to_lsl.py` |
| g.Nautilus via gpype | Parameterized | Parameterized | Varies | Varies | `scripts/bridges/gpype_to_lsl.py` |
| Galaxy Watch | `Smartwatch_IMU`, `Smartwatch_PPG` | `IMU`, `PPG` | 6 (IMU), 2 (PPG) | 50 Hz (IMU), 1 Hz (PPG) | `integrations/galaxy_watch_lsl` -> `scripts/bridges/smartwatch_lsl_bridge.py` |
| Artinis fNIRS | `Artinis_fNIRS` | `fNIRS` | Configurable | Configurable | `scripts/bridges/artinis_to_lsl.py` |
| Task Markers | Varies | `Markers` | 1 | Irregular (0) | `scripts/tasks/common/base_task.py` |
| Games / External Engines | Varies | `Markers` | 1 | Irregular (0) | UDP to LSL bridge via `external_game_engines` |

*Note: The Galaxy Watch smartwatch bridge specifically maps the watch's sensor timestamp onto LSL time (`DeviceClockMapper`), smoothing out jitter from UDP transmission.*

## Adding a New Device

When extending the system with a new hardware device, follow these architectural constraints:

1. **Stream Naming:** Ensure clear and consistent stream names.
2. **Channel Labels & Units:** Use `desc()` to inject an XML definition of channel names and types. Standard format is `.child("channels").child("channel")` where `label` and `type` are defined.
3. **Stable `source_id`:** Guarantee the `source_id` within the `StreamInfo` is stable (e.g., combining the device serial number) to prevent duplicate streams during connection drops.
4. **Push Real Acquisition Timestamps:** When calling `push_sample()` or `push_chunk()`, supply the real hardware acquisition timestamp adjusted to LSL time if available, rather than using `pylsl.local_clock()` at arrival time.
