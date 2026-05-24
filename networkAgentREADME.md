# RoboRanger

RoboRanger is a hardware-based eco tour guide built for DataHacks 2026. Point it at anything in nature and it tells you what it's looking at — in the voice of a national park ranger.

The device observes the environment through a camera, identifies plants and animals using an on-device vision model, detects where it is in the world, and narrates what it sees in conversational, place-specific natural history. Visitors can also ask free-form questions and get answers that weave in everything the ranger has already shown them on the tour.

The demo target is **La Jolla Cove**, but the ranger agent is location-general: it detects its surroundings wherever it's deployed and adapts its narration accordingly.

Target hardware: **Arduino UNO Q** (Linux Debian + Python 3 on a quad-core SoC alongside the classic MCU).

---

## Architecture overview

RoboRanger has two independent workstreams that feed the same product:

```
Camera frame
    │
    ▼
Species Classifier          ← DINOv2 ViT-B/14 fine-tuned on La Jolla iNaturalist data
    │
    ▼ (scientific name + confidence)
Tour Guide Agent            ← Claude Haiku, stateful ranger persona, location-aware
    │
    ▼ (narration text)
Voice I/O                   ← ElevenLabs TTS (speak) + STT (listen)
```

---

## Species classifier

The classifier is the "eyes" of the device. It is a **DINOv2 ViT-B/14 backbone** with a lightweight linear head, trained specifically on species commonly observed at La Jolla Cove.

### Training data

Source: an iNaturalist export (`observations-711999.csv`) for the La Jolla Cove area — roughly 40 species plus a `seashore` negative class for scenery-only frames (waves, cliffs, empty sand). The top 50 most common species with at least 15 images each were kept. This makes the model a local expert rather than a general classifier — that is intentional.

### Training pipeline

Two notebooks run in sequence:

1. **`data_preparation.ipynb`** — reads the CSV, rewrites iNaturalist URLs to the `large` size variant, parallel-downloads images into a cache directory, performs a stratified 70/15/15 train/val/test split per species, and writes everything into `data/{train,val,test}/<Scientific_name>/` — the `ImageFolder` layout PyTorch expects.

2. **`species_identification.ipynb`** — attaches a `LayerNorm → Linear` head to a frozen DINOv2 ViT-B/14 backbone, trains with AdamW + cosine learning rate schedule + mixed-precision (AMP, CUDA only) + early stopping. Setting `FREEZE_BACKBONE = False` switches from linear probing to full fine-tuning. Artifacts are written to `outputs/`: the checkpoint (`best_model.pth`) carries `class_names`, `image_size`, and `model_name` metadata so inference always reconstructs the exact architecture and class order.

### Inference

`classifier.py` loads the checkpoint, rebuilds the DINOv2 head from the saved metadata, and exposes a `predict(image, top_k)` method returning softmax probabilities over the class list. A `CONFIDENCE_THRESHOLD` of 0.7 is applied at demo time — frames below that threshold (or whose top prediction is the `seashore` negative class) are buffered silently rather than narrated.

The classifier can run locally on the device or be offloaded to a remote HTTP endpoint (a Modal deployment) — `load_classifier()` picks based on the `CLASSIFIER_URL` environment variable.

---

## Tour guide agent

The ranger is powered by **Claude Haiku** (`claude-haiku-4-5-20251001`) via the Anthropic API.

### System prompt and persona

A detailed system prompt defines the ranger's behavior across four modes:

- **New sighting** — two-part narration: ~150 words of vivid natural history tied to the specific location and season, followed by ~60 words on concrete, place-specific threats to the species.
- **Repeat sighting** — brief acknowledgment; asks if the visitor wants to hear it again. No re-narration.
- **Free-form question** — conversational 2–4 sentence answer, weaving in species already discussed when genuinely relevant.
- **Unidentified image** — comments briefly on the scene without inventing a species.

The system prompt is sent with `cache_control: ephemeral` on every API call to keep prompt-caching costs low on repeated turns.

### Stateful session (`TourSession`)

`TourSession` maintains a running `messages` list across turns — standard multi-turn conversation history passed to the Claude API. It tracks every species seen by genus+species key (subspecies collapsed) so it can reliably distinguish new sightings from repeats.

Three verbs:
- `see(observation)` — a classified sighting. The session narrates or acknowledges a repeat.
- `ask(text)` — a free-form question answered in the full tour context.
- `look_at(image_path)` — buffer a low-confidence image silently; it is attached as an image block on the next `see()` or `ask()` with a note that the classifier didn't recognize it.

### Location awareness

At session start, `location.py` resolves the device's position through a priority chain:

1. Explicit `lat`/`lon` arguments
2. Attached GPS module via serial (NMEA sentences, `pyserial` + `pynmea2`)
3. IP geolocation via `ipapi.co`

Coordinates are then reverse-geocoded to a human-readable place name via **OpenStreetMap Nominatim** (free, no API key). The result — display name, coordinates, region, country — is injected as a second system block on every API call, grounding the ranger's narration in the actual location regardless of where the device is deployed.

---

## Voice I/O

`voice.py` wraps **ElevenLabs** for both directions of voice interaction.

### Speech-to-text (visitor → ranger)

`VoiceIO.listen()` records the microphone at 16 kHz mono PCM using `sounddevice`, captures until the user presses Enter, then POSTs the WAV buffer to ElevenLabs **Scribe** (`scribe_v1`) for transcription. Recordings shorter than 0.5 seconds are dropped.

### Text-to-speech (ranger → visitor)

`VoiceIO.speak(text)` sends the ranger's narration to ElevenLabs TTS (`eleven_turbo_v2_5`) using the **"Brian"** voice preset — a deep American male that fits the ranger persona. The response is raw PCM at 22.05 kHz, played back via `sounddevice`.

Playback is interruptible: on Windows, `msvcrt.kbhit()` polls for any keypress; on Unix, `select()` watches stdin for Enter. Any keypress stops audio mid-sentence and returns control to the REPL, so the visitor can cut the ranger off immediately to ask a new question or show a new image.

### Cloud deployment

`server.py` exposes the full pipeline as a **FastAPI** HTTP server, letting a lightweight Arduino client offload all heavy work to the cloud. Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session` | Create a new tour session; returns a `session_id` |
| `POST` | `/see` | Upload an image; classifier runs and ranger narrates |
| `POST` | `/ask` | Send a text question to the ranger |
| `POST` | `/look` | Buffer an unidentified image silently |
| `POST` | `/tts` | Convert text to speech via ElevenLabs; returns WAV |
| `POST` | `/stt` | Transcribe audio via ElevenLabs Scribe; returns text |
| `GET`  | `/health` | Liveness check |

---

## Running it

### Training the classifier

```bash
# In Jupyter / VS Code, run in order:
species_identification/data_preparation.ipynb
species_identification/species_identification.ipynb
```

Dependencies: `pandas`, `requests`, `Pillow`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`.

### Running the demo

```bash
pip install -r TourGuide_Agent/requirements.txt

# Create TourGuide_Agent/.env with:
# ANTHROPIC_API_KEY=...
# ELEVEN_LABS_API_KEY=...

python TourGuide_Agent/demo.py              # voice + text REPL
python TourGuide_Agent/demo.py --no-voice   # text only
python TourGuide_Agent/demo.py image.jpg    # open with a first sighting
```

### Running the cloud server

**Terminal 1 — server:**

```bash
cd ~/Desktop/Datahacks2026/TourGuide_Agent
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — ngrok tunnel:**

```bash
ngrok http 8000
```

**Arduino UNO Q:**

```bash
sudo SERVER_URL=https://<ngrok-url> python3 arduino_client.py --camera 2 --mic-device hw:0,0
```

## Key design decisions

**Narrow classifier, wide agent.** The species model is intentionally a La Jolla Cove specialist — trained only on what lives there. The ranger agent is location-general, grounding its narration wherever the device lands via live GPS/IP detection. This keeps training tractable while making the demo compelling and realistic.

**`seashore` as a negative class.** Scenery-only frames (waves, cliffs, empty sand) are a real class during training, not filtered in pre-processing. This teaches the model to express genuine uncertainty rather than confabulating a species for a background shot.

**Confidence gating before the LLM.** The 0.7 softmax threshold and the negative-class check happen before Claude is ever called. Low-confidence frames are buffered silently via `look_at()` — the ranger stays quiet until the visitor asks, rather than narrating something the device isn't sure about.

**Prompt caching.** The ranger system prompt (the longest, most stable token sequence) is marked `cache_control: ephemeral` on every turn, reducing cost and latency on the repeated API calls a multi-turn tour generates.

**Interruptible TTS.** The visitor can cut the ranger off mid-sentence with any keypress. This is essential for a live demo — people want to ask follow-up questions the moment something catches their interest, not wait for a paragraph to finish.
