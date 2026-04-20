"""Lightweight client for the Arduino UNO Q (or any Linux device with a USB camera).

Captures frames from the webcam and sends them to the cloud EcoGuide API.
The server handles all heavy work (classification, narration, TTS).

Usage:
    SERVER_URL=http://<cloud-ip>:8000 python arduino_client.py
    SERVER_URL=http://<cloud-ip>:8000 python arduino_client.py --no-camera

Commands at the >> prompt:
    <Enter>        — capture a frame from the webcam and send to the ranger
    see <path>     — send a specific image file
    ask <text>     — ask the ranger a question
    listen / talk  — record from the microphone, transcribe, and ask the ranger
    quit           — exit

Audio playback: if the server returns audio, it is played via `aplay` (ALSA,
available on Debian/Ubuntu). Falls back to printing text only if audio
playback is not available.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURES_DIR = Path.home() / "captures"

NGROK_HEADERS = {"ngrok-skip-browser-warning": "1"}


def create_session(server_url: str, lat: float | None = None, lon: float | None = None, location: str | None = None) -> str:
    """Create a tour session on the server. Returns session_id."""
    data = {}
    if lat is not None:
        data["lat"] = lat
    if lon is not None:
        data["lon"] = lon
    if location:
        data["location"] = location

    resp = requests.post(f"{server_url}/session", data=data, headers=NGROK_HEADERS, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    print(f"Session created: {result['session_id']}")
    print(f"Location: {result['location']}")
    if result.get("lat"):
        print(f"Coords: ({result['lat']}, {result['lon']})")
    return result["session_id"]


def send_image(server_url: str, session_id: str, image_path: Path) -> dict:
    """POST an image to /see and return the server response."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{server_url}/see",
            files={"image": (image_path.name, f, "image/jpeg")},
            data={"session_id": session_id},
            headers=NGROK_HEADERS,
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


def send_question(server_url: str, session_id: str, text: str) -> dict:
    """POST a text question to /ask and return the server response."""
    resp = requests.post(
        f"{server_url}/ask",
        data={"session_id": session_id, "text": text},
        headers=NGROK_HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def capture_frame(cap) -> Path | None:
    """Capture a single frame from the webcam, save as JPEG, return the path."""
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    import cv2
    cv2.imwrite(tmp.name, frame)
    tmp.close()
    return Path(tmp.name)


def _find_audio_player() -> list[str] | None:
    """Return the command prefix for a usable WAV audio player, or None."""
    for cmd in (
        ["aplay"],           # ALSA — standard on Linux (Arduino UNO Q)
        ["afplay"],          # macOS
        ["mpv", "--no-video"],
        ["ffplay", "-nodisp", "-autoexit"],
    ):
        if shutil.which(cmd[0]):
            return cmd
    return None


def fetch_and_play_audio(server_url: str, text: str) -> None:
    """POST text to /tts, save the WAV response, and play it.

    Playback is interruptible: press Enter to stop audio and return to
    the prompt immediately.
    """
    player = _find_audio_player()
    if player is None:
        return

    try:
        resp = requests.post(
            f"{server_url}/tts",
            data={"text": text},
            headers=NGROK_HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  (TTS unavailable: {e})")
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(resp.content)
    tmp.close()

    print("  [playing — press Enter to interrupt]")
    proc = subprocess.Popen(
        [*player, tmp.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        while proc.poll() is None:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print("  [interrupted]")
                break
    except Exception:
        proc.kill()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _find_mic_device() -> str | None:
    """Auto-detect an ALSA capture device by parsing `arecord -l`."""
    if not shutil.which("arecord"):
        return None
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None

    for line in out.splitlines():
        m = re.match(r"card\s+(\d+):.*device\s+(\d+):", line)
        if m:
            return f"hw:{m.group(1)},{m.group(2)}"
    return None


def record_mic(device: str) -> Path | None:
    """Record from the mic via arecord until Enter is pressed. Returns WAV path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", tmp.name]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("  [recording — press Enter to stop]", flush=True)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    wav = Path(tmp.name)
    if wav.stat().st_size < 1000:
        wav.unlink(missing_ok=True)
        print("  Recording too short.")
        return None
    return wav


def send_audio(server_url: str, wav_path: Path) -> str:
    """POST a WAV file to /stt and return the transcript."""
    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{server_url}/stt",
            files={"audio": (wav_path.name, f, "audio/wav")},
            headers=NGROK_HEADERS,
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json().get("text", "")


def handle_see_response(result: dict, *, server_url: str = "", audio: bool = False) -> None:
    """Print the server's /see response and optionally play TTS audio."""
    if result.get("top3"):
        print("  Top-3 predictions:")
        for p in result["top3"]:
            common = f" — {p['common_name']}" if p.get("common_name") else ""
            print(f"    {p['probability']:6.1%}  {p['scientific_name']}{common}")

    if not result.get("detected"):
        reason = result.get("reason", "unknown")
        if reason == "negative_class":
            print(f"\n  No organism detected (top prediction is a scenery class).")
        elif reason == "low_confidence":
            conf = result.get("confidence", 0)
            print(f"\n  No organism detected (confidence {conf:.1%} below threshold).")
        else:
            print(f"\n  Not detected: {reason}")
        print("  The ranger is waiting. Ask a question about the scene if you'd like.\n")
        return

    species = result.get("species", "unknown")
    common = result.get("common_name", "")
    conf = result.get("confidence", 0)
    display = f"{common} ({species})" if common else species
    print(f"\n  Detected: {display} ({conf:.1%})\n")

    text = result.get("text", "")
    if text:
        print(text)
        print()
        if audio and server_url:
            fetch_and_play_audio(server_url, text)


def main():
    parser = argparse.ArgumentParser(description="EcoGuide Arduino Client")
    parser.add_argument("--no-camera", action="store_true", help="Disable webcam (image files only)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0, try 1 or 2 if wrong device)")
    parser.add_argument("--no-audio", action="store_true", help="Disable TTS audio playback")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone input")
    parser.add_argument("--mic-device", type=str, default=None, help="ALSA capture device (e.g. hw:1,0)")
    parser.add_argument("--server", default=SERVER_URL, help=f"Server URL (default: {SERVER_URL})")
    parser.add_argument("--lat", type=float, default=None, help="Latitude for the tour")
    parser.add_argument("--lon", type=float, default=None, help="Longitude for the tour")
    parser.add_argument("--location", type=str, default=None, help="Location name for the tour")
    args = parser.parse_args()

    server_url = args.server.rstrip("/")
    use_audio = not args.no_audio

    # Health check
    try:
        resp = requests.get(f"{server_url}/health", headers=NGROK_HEADERS, timeout=5)
        resp.raise_for_status()
        print(f"Connected to server: {server_url}")
    except Exception as e:
        print(f"Cannot reach server at {server_url}: {e}")
        print("Make sure the server is running: uvicorn server:app --host 0.0.0.0 --port 8000")
        return 1

    session_id = create_session(server_url, lat=args.lat, lon=args.lon, location=args.location)
    print()

    cap = None
    if not args.no_camera:
        try:
            import cv2
            cap = cv2.VideoCapture(args.camera)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
                print(f"Camera ready ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            else:
                print("Could not open camera. Use 'see <path>' to send image files.")
                cap = None
        except ImportError:
            print("OpenCV not installed. Use 'see <path>' to send image files.")
            cap = None

    mic_device = None
    if not args.no_mic:
        mic_device = args.mic_device or _find_mic_device()
        if mic_device:
            print(f"Microphone ready ({mic_device})")
        else:
            print("No microphone detected. Use 'ask <text>' for typed questions.")

    print()
    print("Commands:")
    print("  <Enter>        — capture frame from camera and send to ranger")
    print("  see <path>     — send an image file")
    print("  ask <text>     — ask the ranger a question")
    if mic_device:
        print("  listen / talk  — speak a question into the mic")
    print("  quit           — exit")
    print()

    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            if cap is None:
                print("No camera available. Use 'see <path>' instead.")
                continue
            print("Capturing frame...")
            frame_path = capture_frame(cap)
            if frame_path is None:
                continue
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            saved = CAPTURES_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            shutil.copy2(frame_path, saved)
            print(f"Saved to {saved}")
            print(f"Sending {frame_path.name} to server...")
            try:
                result = send_image(server_url, session_id, frame_path)
                handle_see_response(result, server_url=server_url, audio=use_audio)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                frame_path.unlink(missing_ok=True)
                line = "listen"
            # continue

        if line.lower() in {"quit", "exit"}:
            break

        if line.lower().startswith("see "):
            path_str = line[4:].strip()
            image_path = Path(path_str)
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            print(f"Sending {image_path.name} to server...")
            try:
                result = send_image(server_url, session_id, image_path)
                handle_see_response(result, server_url=server_url, audio=use_audio)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line.lower().startswith("ask "):
            question = line[4:].strip()
            if not question:
                print("Usage: ask <your question>")
                continue
            try:
                result = send_question(server_url, session_id, question)
                text = result.get("text", "")
                if text:
                    print()
                    print(text)
                    print()
                    if use_audio:
                        fetch_and_play_audio(server_url, text)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line.lower() in {"listen", "talk", "voice"}:
            if mic_device is None:
                print("No microphone available. Use 'ask <text>' instead.")
                continue
            print("  Entering voice mode (say nothing or press Enter with silence to exit)\n")
            while True:
                wav_path = record_mic(mic_device)
                if wav_path is None:
                    print("  Exiting voice mode.\n")
                    break
                try:
                    print("  Transcribing...")
                    transcript = send_audio(server_url, wav_path)
                    if not transcript:
                        print("  (no speech detected — exiting voice mode)\n")
                        break
                    print(f"  You: {transcript}")
                    result = send_question(server_url, session_id, transcript)
                    text = result.get("text", "")
                    if text:
                        print()
                        print(f"  Ranger: {text}")
                        print()
                        if use_audio:
                            fetch_and_play_audio(server_url, text)
                    print("  (speak your next question, or press Enter with silence to exit)")
                except Exception as e:
                    print(f"Error: {e}")
                    break
                finally:
                    wav_path.unlink(missing_ok=True)
            continue

        # Anything else is treated as a question
        try:
            result = send_question(server_url, session_id, line)
            text = result.get("text", "")
            if text:
                print()
                print(text)
                print()
                if use_audio:
                    fetch_and_play_audio(server_url, text)
        except Exception as e:
            print(f"Error: {e}")

    if cap is not None:
        cap.release()
    print("Tour ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
