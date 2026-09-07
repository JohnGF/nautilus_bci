# External Game Engines Integration

This folder contains tools and instructions for integrating external game engines (such as Godot, Unity, or Unreal Engine) with the Nautilus BCI BIDS recording suite.

Because many game engines do not have native, out-of-the-box support for the Lab Streaming Layer (LSL), we provide a lightweight **UDP to LSL Bridge**. This allows any game engine to send simple network messages over UDP, which are then translated into LSL markers and perfectly synchronized with the EEG data in the BIDS dataset.

## 1. Running the UDP to LSL Bridge

The bridge script is a standalone Python application that listens for incoming UDP messages and broadcasts them as LSL markers formatted for the BIDS recorder.

To start the bridge, run:

```bash
uv run python udp_to_lsl_bridge.py
```

By default, it listens on `127.0.0.1:9000`. You can configure the port and the LSL stream name:

```bash
uv run python udp_to_lsl_bridge.py --port 9000 --marker-name ExternalGameMarkers
```

## 2. Sending Markers from Your Game Engine

Your game engine simply needs to send a UDP packet containing a JSON string to the bridge's IP and port.

The expected JSON format is:
```json
{
  "name": "Jump_Action",
  "duration": 0.5
}
```
- `name`: (String) The event name you want recorded in your BIDS `_events.tsv` file.
- `duration`: (Float) The duration of the event in seconds. Use `0.0` for instantaneous events.

### Game Engine Specific Guides

If you are using **Godot 4**, we have provided a ready-to-use GDScript file and specific integration instructions.
👉 **[See the Godot Integration Guide](./godot/README.md)**

## 3. Recording the Data

The component responsible for capturing these LSL streams and saving them into the standardized BIDS format is the **Master BIDS Recorder**. This logic is housed primarily in `scripts/recorders/bids_recorder.py` (and integrated via `multimodal_bids_recorder.py`).

1. Start the main `run_bci_suite.py` application (which hosts the master BIDS recorder).
2. Start the `udp_to_lsl_bridge.py`.
3. In the BCI suite, start the EEG streamer and begin a **BIDS Recording**. The master recorder will automatically detect the "ExternalGameMarkers" LSL stream.
4. Launch your external game (e.g., in Godot) and play.
5. The master recorder will save the data to the standard BIDS directory path, typically located at:
   `scripts/bids_dataset_multimodal/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-XYZ_events.tsv`
   Your external game markers will appear in this file perfectly synchronized alongside the recorded EEG data!
