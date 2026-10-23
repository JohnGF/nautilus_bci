# Quickstart

This guide explains how to start a minimal multimodal recording session with the Nautilus BCI suite.

## 1. Start Bridges

Begin by launching the bridges for your hardware devices. This makes their streams available over LSL. You can run these commands from the root directory or `scripts/` directory.

In a terminal, start a bridge (for example, the Smartwatch in test mode):
```bash
cd scripts
uv run python bridges/smartwatch_lsl_bridge.py --mock
```

To run the bridge for a real smartwatch:
```bash
cd scripts
uv run python bridges/smartwatch_lsl_bridge.py --mode udp --port 5005
```

To add EEG, open a new terminal and run:
```bash
cd scripts
uv run python bridges/gds_to_lsl.py
```

*(If you are setting up the network for the first time, check the [LSL Network Setup](guides/lsl_network_setup.md) to ensure streams are not silently blocked).*

## 2. Verify Streams

Before recording, confirm that the system successfully detects all streams. The recorder engine can discover and connect to streams natively.

Open another terminal and verify discovery:
```bash
cd scripts
uv run python -c "
import sys; sys.path.insert(0, 'recorders')
from multimodal_bids_recorder import MultimodalBIDSRecorder
MultimodalBIDSRecorder().discover_and_connect_streams(timeout=3)"
```

You should see an output listing every available stream (e.g. `Smartwatch_IMU`, `Smartwatch_PPG`, `gNautilus_EEG`).

## 3. Record a Session

To manage the recording of a session directly, use the Master Control Dashboard. The dashboard lets you inspect live signals and launch task paradigms simultaneously.

```bash
cd scripts
uv run python run_bci_suite.py
```

From the dashboard, you can initiate a recording session, which will automatically serialize all active streams into clean BIDS format in the output directory.
