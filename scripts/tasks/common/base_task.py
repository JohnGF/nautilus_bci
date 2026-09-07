import time
from PySide6.QtWidgets import QMainWindow

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

class BaseTaskApp(QMainWindow):
    def __init__(self, marker_name="TaskMarkers", source_id="Task_Markers_2026"):
        super().__init__()
        self.marker_name = marker_name
        self.source_id = source_id

        self.outlet = None
        self.init_lsl()

    def init_lsl(self):
        if HAS_LSL:
            try:
                info = StreamInfo(
                    name=self.marker_name,
                    type='Markers',
                    channel_count=1,
                    nominal_srate=0,
                    channel_format='string',
                    source_id=self.source_id
                )
                self.outlet = StreamOutlet(info)
                print(f"[+] LSL Marker Outlet created successfully ('{self.marker_name}').")
            except Exception as e:
                print(f"[-] Failed to create LSL Marker Outlet: {e}")
                self.outlet = None
        else:
            print("[!] PyLSL not installed. Running in standalone visual mode.")

    def send_marker(self, marker_str, duration=0.1):
        timestamp = local_clock() if HAS_LSL else time.time()
        # Append duration suffix for BIDS recorder parsing
        lsl_str = f"{marker_str}_dur_{duration}"
        print(f"[MARKER @ {timestamp:.3f}] {marker_str} (Duration: {duration}s)")

        # Dispatch to LSL network
        if self.outlet:
            self.outlet.push_sample([lsl_str], timestamp)

        # Hook directly into local recorder if active
        if hasattr(self, 'recorder_widget') and self.recorder_widget and self.recorder_widget.recorder and self.recorder_widget.recorder.is_recording:
            rel_time = timestamp - self.recorder_widget.recorder.start_time_lsl
            self.recorder_widget.recorder.marker_events.append((rel_time, str(marker_str), duration))
