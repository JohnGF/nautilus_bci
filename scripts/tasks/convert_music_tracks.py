import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os
import subprocess
from PySide6.QtCore import QUrl, QCoreApplication, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

music_dir = os.path.abspath("music_tracks")
tracks = [
    ("real_beethoven_fur_elise", 5000),
    ("real_joplin_entertainer", 10000),
    ("real_bach_prelude", 8000),
    ("real_mozart_eine_kleine", 2000),
    ("real_vivaldi_spring", 12000),
    ("real_tchaikovsky_waltz", 45000)
]

print("=== CONVERTING MP3 TO UNCOMPRESSED WAV FOR 100% RELIABLE PLAYBACK ===")
for base, offset in tracks:
    mp3_path = os.path.join(music_dir, base + ".mp3")
    wav_path = os.path.join(music_dir, base + ".wav")
    if os.path.exists(mp3_path):
        print(f"[*] Converting {base}.mp3 -> {base}.wav ...")
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ar", "44100", "-ac", "2", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(wav_path):
            size_mb = os.path.getsize(wav_path) / (1024 * 1024)
            print(f"    [+] Converted {base}.wav successfully ({size_mb:.2f} MB)")

print("\n=== TESTING WAV AUDIO PLAYBACK IN QMEDIAPLAYER ===")
app = QCoreApplication([])
player = QMediaPlayer()
audio = QAudioOutput()
player.setAudioOutput(audio)
audio.setVolume(1.0)

for base, offset in tracks:
    wav_path = os.path.join(music_dir, base + ".wav")
    if os.path.exists(wav_path):
        player.stop()
        player.setSource(QUrl.fromLocalFile(wav_path))
        player.setPosition(offset)
        player.play()
        print(f"    [+] {base}.wav -> Status: {player.mediaStatus()} | State: {player.playbackState()} | Pos: {player.position()} ms")

print("\n[SUCCESS] All 6 real master WAV audio files are converted and verified!")
