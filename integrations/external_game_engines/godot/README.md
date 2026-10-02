# Godot Engine Integration Guide

This folder provides a ready-to-use GDScript to help you send event markers from a Godot 4 game to the Nautilus BCI suite.

## The Script: `bci_marker_sender.gd`

The `bci_marker_sender.gd` script is a simple Node that opens a UDP socket and sends JSON formatted messages. These messages are caught by the Python `udp_to_lsl_bridge.py` (located in the parent directory) and injected into your BIDS recording.

## How to use in your Godot Project

The easiest way to integrate this is to add the script as a globally accessible **Autoload (Singleton)** in your Godot project.

1. **Copy the script**
   Copy the `bci_marker_sender.gd` file into your Godot project folder (e.g., into a `scripts/` folder).

2. **Add it as an Autoload**
   - In the Godot editor, go to **Project** -> **Project Settings**.
   - Navigate to the **Autoload** tab.
   - Click the folder icon next to the "Path" field, select your copied `bci_marker_sender.gd` file.
   - Name the node `BCIMarkers` and click **Add**.

3. **Send markers from anywhere**
   Because you added it as an Autoload named `BCIMarkers`, you can now send BIDS markers from *any other script* in your entire game just by calling the function:

   ```gdscript
   extends CharacterBody3D

   func jump():
       velocity.y = JUMP_VELOCITY

       # Send an instantaneous marker to the BIDS recorder
       BCIMarkers.send_bci_marker("Player_Jump", 0.0)

   func take_damage():
       # Send a marker with a 1.5 second duration (e.g., for a recovery animation)
       BCIMarkers.send_bci_marker("Player_Damaged", 1.5)
   ```

## Testing the Connection

1. Run the python bridge from the parent directory: `uv run python udp_to_lsl_bridge.py`
2. Run your Godot game.
3. When the game starts, the `_ready()` function in the script will automatically send a `Game_Started` marker.
4. Check the terminal window where the python bridge is running. You should see a message confirming the `Game_Started` marker was received and forwarded to LSL.
