# Wearable Protocol (W2)

W2 is the UDP packet format that lets any number of wearables stream to one bridge at the same time: Wear OS watches, ESP32 boards, or anything else that can send UDP over Wi-Fi. A typical setup is both wrists and both ankles. The bridge (`scripts/bridges/smartwatch_lsl_bridge.py`) gives every device its own LSL streams and its own clock mapping.

```bash
uv run python bridges/smartwatch_lsl_bridge.py --mode udp --port 5005
uv run python bridges/smartwatch_lsl_bridge.py --mock   # four simulated devices, no hardware
```

## Packet

One UDP datagram, ASCII, at most 1400 bytes:

```text
W2|<device_id>|<location>|<seq>|<kind>[@<rate_hz>]|<t_ns>,<v1>,<v2>,...;<t_ns>,<v1>,...
```

| Field | Meaning |
|---|---|
| `device_id` | Stable per physical device, 1–32 chars of `A-Z a-z 0-9 _ -`. Becomes the LSL `source_id`, so a recorder reconnects to the same stream after a device restart. |
| `location` | Body site, same character set. Use `L_wrist`, `R_wrist`, `L_ankle`, `R_ankle` where they apply. Becomes part of the stream name. |
| `seq` | Packet counter per device, across all kinds, wrapping at 2³². Gaps count as packet loss. A jump of 1000 or more is treated as a device restart. |
| `kind` | What the samples are; see below. The optional `@rate` sets the LSL nominal rate (default 0, irregular). |
| samples | One or more, separated by `;`. Each starts with `t_ns`, then the channel values. All samples in one packet have the same channel count. |
| `t_ns` | Acquisition time in nanoseconds on the device's own monotonic clock. Every kind from one device must use the same clock. |

### Kinds

| Kind | Channels | LSL stream | LSL type |
|---|---|---|---|
| `IMU` | accel x,y,z (m/s²), gyro x,y,z (rad/s) | `IMU_<location>` | `IMU` |
| `HR` | heart rate (bpm) | `HR_<location>` | `PPG` |
| `TOUCH` | x, y (px), touched (0/1) | `TOUCH_<location>` | `Touch` |
| `HB` | sent, dropped, battery % | none, status only | — |
| `ECHO` | ping id, turnaround µs (see [Timing](#echo)) | `ClockSync_<location>` (raw exchanges) | `ClockSync` |

Every data stream also gets a last channel, `DeviceTime`: the sample's raw device clock in seconds. Data streams are `double64` so it keeps full precision. Consumers that read the first N channels by position are unaffected.
| anything else | `ch1..chN` | `<kind>_<location>` | `<kind>` |

Send `HB` once a second. It drives the bridge's status table (battery level, stale-device warning) and keeps the clock mapping fresh when the IMU is idle.

### Example

```text
W2|wear-3fa91c|L_wrist|8812|IMU@50|51234560000000,0.12,-0.40,9.79,0.01,0.00,-0.02;51234580000000,0.10,-0.41,9.80,0.01,0.01,-0.02
```

## Timing

LSL keeps clocks in sync between machines that run liblsl, but a watch or an ESP32 sends plain UDP, so the device-to-bridge hop is outside LSL. The bridge closes that gap with ping/echo, the same method NTP and LSL's own `time_correction` use.

```text
bridge  t1 ── W2P|<id> ──▶  t2  device
bridge  t4 ◀── ECHO ────────  t3  device

offset = ((t1 − t2) + (t4 − t3)) / 2     LSL time − device time
delay  = (t4 − t1) − (t3 − t2)           network round trip
```

- **Exact if both legs take equally long**, and wrong by at most `delay / 2` otherwise. That is the `±` the status table shows for each device. It is a worst case; in simulation with 1–15 ms of jitter per leg, the real error stayed under 1.5 ms.
- **Bursts, best exchange only:** 40 pings when a device appears, then 5 every 10 s. Wi-Fi delay varies a lot, so each burst keeps only its fastest exchange.
- **Drift:** device and PC crystals disagree by tens of ppm, which is 36–180 ms per hour. A straight line (offset + drift) is fitted through the last 30 bursts (~5 min), so one handshake at the start would not be enough.
- **Pings ride the device's own traffic:** the bridge sends a ping right after a packet from that device arrives, while its radio is still awake. Otherwise Wi-Fi power save would hold the ping back until the next beacon.
- **Raw exchanges are recorded:** each device gets a `ClockSync_<location>` stream (`double64`: t1, t2, t3, t4), so the fit can be redone offline over the whole session, using exchanges both before and after each sample.
- **Fallback:** until the first burst completes, and for devices that never answer (they are left alone after 3 silent bursts), the bridge uses the smallest `arrival − device_time` of the last 10 s. That includes the one-way network latency, typically a few milliseconds.

Batching samples does not hurt timing, because every sample keeps its own `t_ns`. Batch about 5 IMU samples per packet: at 50 Hz that is 10 packets/s per device instead of 50, which saves battery and Wi-Fi airtime when four devices share one hotspot.

### Echo

A device that supports sync listens on the socket it sends from. For each `W2P|<id>` datagram it replies to the sender with:

```text
W2|<device_id>|<location>|<seq>|ECHO|<t2_ns>,<id>,<turnaround_us>
```

where `t2_ns` is its clock when the ping arrived and `turnaround_us` is the time from then until the reply is sent. Reply directly from the receiving code, not through a send queue, since queueing between the timestamps and the wire adds error.

### Validate it

Do a sync gesture at the start and end of a recording (stamp each foot, clap) and compare where the spike lands in the EEG and in each wearable. This checks the whole chain, including the EEG bridge's own latency, which ping/echo does not see.

## Status table

Every 5 s the bridge prints one line per device:

```text
[status] 4 device(s)
  L_ankle    esp-a1         ok    IMU  50.0Hz        loss  0.0%  batt   82%  seen  0.1s ago  sync ±1.9ms drift +21ppm
  L_wrist    wear-3fa91c    ok    HR    1.0Hz IMU  50.0Hz  loss  0.2%  batt   87%  seen  0.0s ago  sync ±2.4ms drift -8ppm
```

`STALE` means nothing has arrived for 3 s. `sync: pending` means the first burst has not completed; `sync: no echo (fallback)` means the device does not answer pings.

## Minimal ESP32 sender

```cpp
// Arduino-ESP32 sketch (not yet run on hardware): 5-sample IMU batches at 50 Hz, a 1 Hz heartbeat,
// and clock-sync echoes
#include <WiFi.h>
#include <WiFiUdp.h>

WiFiUDP udp;
const char* BRIDGE = "192.168.137.1";
String id = "esp-" + String((uint32_t)ESP.getEfuseMac(), HEX);
uint32_t seq = 0;

void sendPacket(const String& kind, const String& body) {
  String p = "W2|" + id + "|L_ankle|" + String(seq++) + "|" + kind + "|" + body;
  udp.beginPacket(BRIDGE, 5005);
  udp.print(p);
  udp.endPacket();
}

void answerPings() {
  // Clock sync: reply to W2P|<id> straight away with arrival time and turnaround
  while (int len = udp.parsePacket()) {
    uint64_t t2 = (uint64_t)esp_timer_get_time() * 1000ULL;
    char buf[32];
    int n = udp.read(buf, sizeof(buf) - 1);
    buf[n > 0 ? n : 0] = 0;
    if (strncmp(buf, "W2P|", 4) != 0) continue;
    IPAddress from = udp.remoteIP();
    uint16_t port = udp.remotePort();
    uint64_t t3 = (uint64_t)esp_timer_get_time() * 1000ULL;
    String p = "W2|" + id + "|L_ankle|" + String(seq++) + "|ECHO|" + String(t2) + "," +
               String(buf + 4) + "," + String((uint32_t)((t3 - t2) / 1000));
    udp.beginPacket(from, port);
    udp.print(p);
    udp.endPacket();
  }
}

void setup() {
  WiFi.begin("ssid", "password");
  while (WiFi.status() != WL_CONNECTED) delay(100);
  WiFi.setSleep(false);  // keep latency low; costs battery
  udp.begin(5006);       // local port; the bridge replies to whatever port we send from
}

void loop() {
  String body;
  for (int i = 0; i < 5; i++) {
    uint64_t t_ns = (uint64_t)esp_timer_get_time() * 1000ULL;
    float ax = 0, ay = 0, az = 9.81, gx = 0, gy = 0, gz = 0;  // read your IMU here
    if (i) body += ";";
    body += String(t_ns) + "," + ax + "," + ay + "," + az + "," + gx + "," + gy + "," + gz;
    uint32_t until = millis() + 20;
    while (millis() < until) answerPings();
  }
  sendPacket("IMU@50", body);
  static uint32_t lastHb = 0;
  if (millis() - lastHb > 1000) {
    lastHb = millis();
    sendPacket("HB", String((uint64_t)esp_timer_get_time() * 1000ULL) + "," + seq + ",0,100");
  }
}
```

Bluetooth-only boards (e.g. nRF52) cannot send UDP directly. They need a gateway, such as a phone or the PC, that reads them over BLE and forwards W2 packets.

## Legacy packets

The bridge still accepts the old single-watch CSV packet (`accX,...,hr,touchX,touchY,isTouched,timestampNs`) and publishes it as `Smartwatch_IMU` / `Smartwatch_PPG`. Those two streams are only created once a legacy packet arrives.

## Recording

`multimodal_bids_recorder.py` exports, per session:

| Path | Content |
|---|---|
| `motion/*_motion.tsv`, `ppg/*_ppg.tsv`, `touch/*_touch.tsv` | `timestamp_sec`, then the data channels by label (`Accel_X`, …). Same layout the analysis scripts read; `DeviceTime` is never in here. |
| `*_motion.json`, `*_ppg.json`, … | Sampling rate, column units, stream name, LSL `source_id`, `DeviceID`, `TrackedPointLocation`, clock-sync method |
| `motion/*_channels.tsv` | BIDS motion channels: `name`, `component` (x/y/z), `type` (ACCEL/GYRO), `tracked_point`, `units` |
| `sourcedata/…/<datatype>/*_devicetime.tsv` | Live `timestamp_sec` and the raw `device_time_s` of every sample |
| `sourcedata/…/clocksync/*_clocksync.tsv` | Ping/echo exchanges; `t1`/`t4` on the `timestamp_sec` axis, `t2`/`t3` device clock |

With one stream of a kind, the filename is unchanged (`..._motion.tsv`). With several, each file gets a BIDS label from its stream name, e.g. `..._tracksys-IMULwrist_motion.tsv` and `..._recording-HRLwrist_ppg.tsv`.

### Offline clock refit

The bridge's live fit only knows the exchanges seen so far. After the session, `refit_wearable_clocks.py` refits each device using exchanges from both before and after every sample: the fastest exchange per burst, a local offset + drift line over ±150 s, and linear interpolation between bursts.

```bash
uv run python recorders/refit_wearable_clocks.py bids_dataset_multimodal          # report
uv run python recorders/refit_wearable_clocks.py bids_dataset_multimodal --apply  # rewrite timestamp_sec
```

The report gives, per stream, the number of bursts, the device drift (positive = device clock runs fast), the median error bound and how far the refit moves the timestamps. `--apply` rewrites `timestamp_sec` and adds a `ClockRefit` block to the JSON. The live timestamps stay in the devicetime file, so applying again gives the same result. In a simulated 10-minute session (40 ppm drift, 1–15 ms of jitter per leg, live timestamps 8 ms off before the first burst), every refit sample lands within 2 ms of the true time.

The recorder only receives exchanges from after it connects, so start it before the wearables (or at least before the session's first stomp/clap). A session needs at least one burst, which is every 10 s; longer sessions give the drift fit more to work with.
