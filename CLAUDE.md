# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Eco Tour Guide (DataHacks 2026)

A hardware-based eco tour guide. The device observes the user's surroundings (camera), combines visual input with location + seasonality, identifies flora and fauna, and narrates a national-park-ranger-style tour. **The hackathon demo target is La Jolla Cove**, and the species classifier is trained specifically for that locale — but the ranger agent itself is now location-general: it detects the device's current place at session start (GPS → IP fallback + Nominatim reverse geocoding) and grounds narration wherever the device is being used.

Target deployment hardware is the **Arduino UNO Q** (Linux Debian + Python 3 on a quad-core SoC alongside the classic MCU), so any new dependency must be pip-installable on that platform.

The repo has two workstreams that feed the same product:

- **`species_identification/`** — trains the on-device / cloud image classifier. Scope is intentionally narrow: the species commonly observed at La Jolla Cove, sourced from iNaturalist, plus a `seashore` negative class for scenery-only frames. The resulting model is the "eyes" of the tour guide.
- **`TourGuide_Agent/`** — the AI agent layer. Wraps the classifier, the Claude ranger persona, and voice I/O into an interactive tour. See its own section below.

Input data: `observations-711999/observations-711999.csv` — an iNaturalist export for the La Jolla Cove area (id, image_url, scientific_name, common_name, lat/lon, observed_on, etc.). Lat/lon and `observed_on` are the hooks for the agent's location/seasonality reasoning.

## species_identification pipeline: raw CSV → trained model

Two notebooks run in order and hand off via filesystem, not variables:

1. **`data_preparation.ipynb`** reads `../observations-711999/observations-711999.csv`, keeps the top-N most common species at La Jolla Cove (default: top 50, min 15 images/class → ~40 classes), rewrites iNaturalist image URLs to a chosen size variant (`medium` → `large`), parallel-downloads into `images_raw/` (cached — safe to re-run), does a stratified per-species 70/15/15 split, then copies images into `data/{train,val,test}/<Scientific_name>/` — the `ImageFolder` layout the trainer expects.

2. **`species_identification.ipynb`** loads `data/` with `torchvision.datasets.ImageFolder`, attaches a linear head to a frozen DINOv2 ViT-B/14 backbone (`torch.hub.load("facebookresearch/dinov2", ...)`), trains with AdamW + cosine LR + AMP + early stopping, and writes artifacts to `outputs/`:
   - `best_model.pth` — state dict + `class_names`, `image_size`, `model_name` metadata (this is what the agent will load)
   - `class_names.json`, `metrics.json` (full history), `training_curves.png`

Toggling `FREEZE_BACKBONE = False` switches from linear probing to full fine-tuning and automatically uses the lower `LR_FINETUNE` (1e-5 vs 1e-3).

Because the demo is La Jolla Cove only, it's fine — expected, even — that the class list is a small, biased subset (heavily weighted toward sea lions, cormorants, gulls, pelicans). Do not "fix" this imbalance by sampling broader geography; the model is meant to be a local expert, not a general species classifier.

The class list also includes **`seashore`** — a negative / "nothing to narrate" class for scenery frames (waves, cliffs, empty sand). It is a regular class from the trainer's perspective but is treated specially at inference time (see TourGuide_Agent below).

## TourGuide_Agent

Five modules, each with a narrow job:

- **`classifier.py`** — `SpeciesClassifier.load()` rebuilds the DINOv2 architecture from `best_model.pth` metadata (`model_name`, `class_names`, `image_size`), loads the state dict, and exposes `predict(image, top_k)` returning `Prediction(scientific_name, common_name, probability)`. Common names are resolved from `observations-711999.csv`. Device auto-detects CUDA → MPS → CPU.
- **`location.py`** — place detection. `get_location()` resolves (in order) explicit lat/lon args → attached GPS module on `GPS_PORT` (NMEA via optional `pyserial` + `pynmea2`) → IP geolocation via `ipapi.co` → `Location.unknown()`. Reverse-geocodes coordinates to a human-readable `display_name`, `country`, and `region` using OpenStreetMap Nominatim (free, no key, 1 req/sec, identifying `User-Agent` required). All network calls have short timeouts and fall back silently.
- **`tour_guide.py`** — the Claude-powered naturalist ranger. The system prompt is location-general: it takes the current place + coordinates + season from the session context on every turn rather than hard-coding a locale. `narrate(Observation)` is a one-shot helper; the real entry point is `TourSession`, a stateful multi-turn tour that remembers every species shown so far. When constructed without an explicit `location`, the session calls `get_location()` at init, so **there is a one-time network delay when a session starts**. Three verbs:
  - `see(Observation)` — a sighting. The session decides NEW vs REPEAT by genus+species key (subspecies collapsed) and the ranger narrates accordingly.
  - `ask(text)` — free-form question answered in the context of the tour so far.
  - `look_at(image_path)` — buffer an image silently without narration; attached to the next `see()`/`ask()` with a note that the classifier didn't recognize it. Used for low-confidence frames.
  Model defaults to `claude-haiku-4-5-20251001` with the system prompt ephemerally cached. Season is hard-coded to Spring.
- **`voice.py`** — `VoiceIO` wraps ElevenLabs STT (`scribe_v1`) and TTS (`eleven_turbo_v2_5`, preset voice "Brian"). `listen()` records mic until Enter and transcribes; `speak(text)` plays TTS **interruptibly** — polls for a keystroke during playback (msvcrt on Windows, `select()` on Unix), stops audio on interrupt, and returns True so the REPL can move on. Requires `ELEVEN_LABS_API_KEY`.
- **`demo.py`** — interactive REPL that ties it all together. Commands: `see <image>` (runs classifier + narrates), `see <image> "<scientific>" ["<common>"]` (skip the classifier), bare `<Enter>` (voice turn), anything else (text question), `quit`. Loaded env via `python-dotenv` from `TourGuide_Agent/.env`. Prints the detected location at startup.

Key inference-side policies in `demo.py`:

- `CONFIDENCE_THRESHOLD = 0.7` on the top-1 softmax. Below that, the image is routed to `session.look_at()` instead of `session.see()` so the ranger stays quiet until the visitor asks about it.
- `NEGATIVE_CLASSES = {"seashore"}`. If the top-1 prediction is a negative class, it's treated identically to a low-confidence result, regardless of probability. Add new negative classes here if the trainer gains more of them.

When touching the agent, prefer loading the classifier checkpoint via the metadata saved in `best_model.pth` (`class_names`, `image_size`, `model_name`) rather than re-deriving from a directory listing — training and inference must agree on class order.

## Key conventions

- **Class label = `scientific_name.replace(" ", "_")`**. `ImageFolder` sorts alphabetically, so `class_names` order is deterministic and must match between training and inference — always restore class names from the checkpoint.
- **Image size variants** are swapped via URL token substitution in `upgrade_url()` (`/medium.jpg` → `/large.jpg`). iNaturalist supports: `square`, `small`, `medium`, `large`, `original`.
- **`images_raw/` is a download cache** keyed by observation `id`. It is gitignored and can be deleted; `data_preparation.ipynb` will re-download. `data/` is likewise derived and gitignored — never commit either.
- **Device selection** auto-detects CUDA → MPS → CPU. AMP (`torch.amp.autocast`/`GradScaler`) is CUDA-only and controlled by `USE_AMP`.
- **Backbone stays in `eval()` during training** when frozen — the train loop explicitly re-asserts this after `model.train()` to keep norm/dropout layers quiet.
- **Scope discipline**: the *classifier* is a La Jolla specialist by design — do not broaden its dataset or try to make it generic. The *agent / ranger prompt* is intentionally location-general now, so it can plausibly narrate anywhere the device lands, given whatever species the local classifier returns. Keep that split: narrow eyes, wide voice.

## Environment notes

- Developed on Windows with CUDA 11.8, PyTorch 2.6, torchvision 0.21, RTX 4070 Laptop GPU. Shell is bash (Git Bash / WSL-style), so use forward slashes and `/dev/null`.
- `CSV_PATH` in `data_preparation.ipynb` uses a Windows-style relative path (`..\observations-711999\observations-711999.csv`) which Python parses with a `SyntaxWarning: invalid escape sequence`. Harmless but worth fixing to a raw string or forward slashes if touched.
- `.gitignore` references `observations-711989` but the actual directory is `observations-711999` — the ignore rule is stale. The raw/processed image dirs are still correctly ignored via their own entries.

## Running

Two entry points:

1. **Training** — run the notebooks in `species_identification/` top-to-bottom in Jupyter / VS Code: `data_preparation.ipynb` then `species_identification.ipynb`. Notebook-inferred deps: `pandas`, `requests`, `Pillow`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`. No package manifest for the notebook side.
2. **Agent / demo** — `pip install -r TourGuide_Agent/requirements.txt`, create `TourGuide_Agent/.env` with `ANTHROPIC_API_KEY` and `ELEVEN_LABS_API_KEY`, then `python TourGuide_Agent/demo.py [image_path]`. Add `--no-voice` to disable STT/TTS (auto-disabled when stdin isn't a TTY). No tests yet.
