"""Cloud API server for the EcoGuide Tour Guide.

Wraps the species classifier + TourSession into HTTP endpoints so a
lightweight client (e.g. Arduino UNO Q) can offload all heavy work to the
cloud.

Start:
    cd TourGuide_Agent
    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /session          — create a new tour session
    POST /see              — send an image; classifier runs, ranger narrates
    POST /ask              — send a text question to the ranger
    POST /look             — send an unidentified image (buffered silently)
    POST /tts              — convert text to speech via ElevenLabs (returns WAV)
    POST /stt              — transcribe audio via ElevenLabs Scribe (returns text)
    GET  /health           — liveness check
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from classifier import load_classifier
from tour_guide import Location, Observation, TourSession

CONFIDENCE_THRESHOLD = 0.7
NEGATIVE_CLASSES = {"seashore"}

app = FastAPI(title="EcoGuide Tour API", version="1.0.0")

# ── Global state ───────────────────────────────────────────────────────────────

classifier = None
tts_client = None
sessions: dict[str, TourSession] = {}

TTS_VOICE_ID = "nPczCjzI2devNBz1zQrb"  # "Brian" — deep American male, ranger persona
TTS_MODEL = "eleven_turbo_v2_5"
STT_MODEL = "scribe_v1"


@app.on_event("startup")
def startup():
    global classifier, tts_client
    print("Loading species classifier...")
    classifier = load_classifier()
    print(f"Classifier ready on {classifier.device}")

    api_key = os.environ.get("ELEVEN_LABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if api_key:
        from elevenlabs.client import ElevenLabs
        tts_client = ElevenLabs(api_key=api_key)
        print("ElevenLabs TTS + STT ready")
    else:
        print("WARNING: ELEVEN_LABS_API_KEY not set — /tts and /stt endpoints disabled")


def _get_session(session_id: str) -> TourSession:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. POST /session first.")
    return sessions[session_id]


def _save_upload(upload: UploadFile) -> Path:
    """Write an uploaded image to a temp file and return the path."""
    suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return Path(tmp.name)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "classifier_loaded": classifier is not None}


@app.post("/session")
def create_session(
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    location: str | None = Form(None),
):
    """Create a new tour session. Returns a session_id."""
    session_id = uuid.uuid4().hex[:12]

    loc = None
    if lat is not None and lon is not None:
        from location import get_location
        loc = get_location(lat=lat, lon=lon)
    elif location:
        loc = Location(display_name=location)

    kwargs = {}
    if loc:
        kwargs["location"] = loc

    sessions[session_id] = TourSession(**kwargs)

    session_loc = sessions[session_id].location
    return {
        "session_id": session_id,
        "location": session_loc.display_name,
        "lat": session_loc.lat,
        "lon": session_loc.lon,
    }


@app.post("/see")
async def see(
    image: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Send an image to the ranger. Runs the classifier, then narrates.

    If the classifier is not confident enough (below threshold or negative
    class), the image is buffered silently via look_at() and the response
    indicates no species was detected.
    """
    session = _get_session(session_id)
    image_path = _save_upload(image)

    try:
        predictions = classifier.predict(image_path, top_k=3)
    except Exception as e:
        session.look_at(image_path)
        return JSONResponse(content={
            "detected": False,
            "reason": f"classifier_error: {e}",
            "text": None,
            "species": None,
            "common_name": None,
            "confidence": None,
            "top3": [],
        })

    top = predictions[0]
    is_negative = top.scientific_name.lower() in NEGATIVE_CLASSES
    below_threshold = top.probability < CONFIDENCE_THRESHOLD

    top3 = [
        {"scientific_name": p.scientific_name, "common_name": p.common_name, "probability": round(p.probability, 4)}
        for p in predictions
    ]

    if is_negative or below_threshold:
        session.look_at(image_path)
        reason = "negative_class" if is_negative else "low_confidence"
        return JSONResponse(content={
            "detected": False,
            "reason": reason,
            "text": None,
            "species": top.scientific_name,
            "common_name": top.common_name,
            "confidence": round(top.probability, 4),
            "top3": top3,
        })

    obs = Observation(
        image_path=image_path,
        scientific_name=top.scientific_name,
        common_name=top.common_name,
    )
    narration = session.see(obs)

    return JSONResponse(content={
        "detected": True,
        "reason": None,
        "text": narration,
        "species": top.scientific_name,
        "common_name": top.common_name,
        "confidence": round(top.probability, 4),
        "top3": top3,
    })


@app.post("/ask")
async def ask(
    session_id: str = Form(...),
    text: str = Form(...),
):
    """Send a free-form text question to the ranger."""
    session = _get_session(session_id)
    reply = session.ask(text)
    return {"text": reply}


@app.post("/look")
async def look(
    image: UploadFile = File(...),
    session_id: str = Form(...),
):
    """Buffer an image without narrating (unidentified subject)."""
    session = _get_session(session_id)
    image_path = _save_upload(image)
    session.look_at(image_path)
    return {"status": "buffered"}


@app.post("/tts")
async def tts(text: str = Form(...)):
    """Convert text to speech via ElevenLabs. Returns WAV audio."""
    if tts_client is None:
        raise HTTPException(status_code=503, detail="TTS not configured — set ELEVEN_LABS_API_KEY")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    import io, struct

    stream = tts_client.text_to_speech.convert(
        voice_id=TTS_VOICE_ID,
        text=text,
        model_id=TTS_MODEL,
        output_format="pcm_22050",
    )
    pcm = b"".join(stream)

    sample_rate = 22050
    channels = 1
    bits_per_sample = 16
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels,
        sample_rate, sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8, bits_per_sample,
        b"data", data_size,
    )
    return Response(content=header + pcm, media_type="audio/wav")


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """Transcribe audio via ElevenLabs Scribe. Accepts WAV upload."""
    if tts_client is None:
        raise HTTPException(status_code=503, detail="STT not configured — set ELEVEN_LABS_API_KEY")

    import io

    audio_bytes = audio.file.read()
    if len(audio_bytes) < 1000:
        return {"text": ""}

    buf = io.BytesIO(audio_bytes)
    buf.name = audio.filename or "recording.wav"
    result = tts_client.speech_to_text.convert(
        file=buf,
        model_id=STT_MODEL,
    )
    return {"text": (result.text or "").strip()}
