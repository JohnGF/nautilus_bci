extends Node

var udp_peer := PacketPeerUDP.new()
var bridge_host := "127.0.0.1"
var bridge_port := 9000

func _ready():
    # Connect to the local UDP bridge running from the Python suite
    udp_peer.connect_to_host(bridge_host, bridge_port)
    print("BCI UDP Peer connected to ", bridge_host, ":", bridge_port)

    # Send an initial marker to verify connection
    send_bci_marker("Game_Started", 0.0)

func send_bci_marker(marker_name: String, duration: float = 0.0):
    # Construct the JSON payload required by the bridge
    var payload = {
        "name": marker_name,
        "duration": duration
    }

    # Convert dictionary to JSON string
    var json_string = JSON.stringify(payload)

    # Send the packet over UDP
    udp_peer.put_packet(json_string.to_utf8_buffer())
    print("Sent BCI marker: ", marker_name)

# ---------------------------------------------------------
# Example Usage:
# You can call send_bci_marker() from anywhere in your code
# if you set this script up as an Autoload (Singleton).
#
# func _input(event):
#     if event.is_action_pressed("ui_accept"):
#         # Send a marker indicating an event with a 0.5s expected duration
#         send_bci_marker("Player_Action_Confirm", 0.5)
# ---------------------------------------------------------
