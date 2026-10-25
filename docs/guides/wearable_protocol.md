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
| anything else | `ch1..chN` | `<kind>_<location>` | `<kind>` |

Send `HB` once a second. It drives the bridge's status table (battery level, stale-device warning) and keeps the clock mapping fresh when the IMU is idle.

### Example

```text
W2|wear-3fa91c|L_wrist|8812|IMU@50|51234560000000,0.12,-0.40,9.79,0.01,0.00,-0.02;51234580000000,0.10,-0.41,9.80,0.01,0.01,-0.02
```

## Timing

The bridge maps each device's `t_ns` onto LSL time with its own `DeviceClockMapper`: the offset is the smallest `arrival − device_time` seen in the last 10 s. Batching samples does not hurt timing, because every sample keeps its own `t_ns`. Batch about 5 IMU samples per packet: at 50 Hz that is 10 packets/s per device instead of 50, which saves battery and Wi-Fi airtime when four devices share one hotspot.

What remains after mapping is each device's constant minimum network latency, typically a few milliseconds. It differs from device to device, so for tight cross-device alignment do a sync gesture at the start of the session (e.g. stamp both feet and clap).

## Status table

Every 5 s the bridge prints one line per device:

```text
[status] 4 device(s)
  L_ankle    esp-a1         ok    IMU  50.0Hz        loss  0.0%  batt   82%  seen  0.1s ago
  L_wrist    wear-3fa91c    ok    HR    1.0Hz IMU  50.0Hz  loss  0.2%  batt   87%  seen  0.0s ago
```

`STALE` means nothing has arrived for 3 s.

## Minimal ESP32 sender

```cpp
// Arduino-ESP32 sketch (not yet run on hardware): 5-sample IMU batches at 50 Hz plus a 1 Hz heartbeat
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

void setup() {
  WiFi.begin("ssid", "password");
  while (WiFi.status() != WL_CONNECTED) delay(100);
  WiFi.setSleep(false);  // keep latency low; costs battery
}

void loop() {
  String body;
  for (int i = 0; i < 5; i++) {
    uint64_t t_ns = (uint64_t)esp_timer_get_time() * 1000ULL;
    float ax = 0, ay = 0, az = 9.81, gx = 0, gy = 0, gz = 0;  // read your IMU here
    if (i) body += ";";
    body += String(t_ns) + "," + ax + "," + ay + "," + az + "," + gx + "," + gy + "," + gz;
    delay(20);
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

`multimodal_bids_recorder.py` writes one TSV per wearable stream. If only one stream of a kind is present, the filename is unchanged (`..._motion.tsv`). With several, each file gets a BIDS label from its stream name, e.g. `..._tracksys-IMULwrist_motion.tsv` and `..._recording-HRLwrist_ppg.tsv`.
