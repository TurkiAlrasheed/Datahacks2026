# RoboRanger — On-Device Wildlife Guide

A portable AI wildlife guide that runs **fully locally** on the **Arduino UNO Q** (quad-core SoC, Linux Debian). A visitor points the camera at an animal or plant, asks a question out loud, and gets a spoken answer — no internet connection required after setup.

Scoped to **La Jolla Cove** species, but the pipeline is locale-general: swap the classifier and corpus for any location.

Devpost: https://devpost.com/software/roboranger?ref_content=my-projects-tab&ref_feature=my_projects

---

## How It Works

The primary entry point is **`full_roboranger_run.py`**, which integrates all four components into one session. On startup it identifies the species from the camera once, announces what it sees, then enters a push-to-talk Q&A loop for the rest of the session. Two utility scripts cover isolated testing:

- **`full_roboranger_run.py`** ← main entry point. Loads Piper first (so it can speak errors), loads Whisper in a background thread while camera ID runs, classifies one frame, announces the species, then loops on push-to-talk questions.
- **`live_inference_mv3.py`** — headless camera loop only. Classifies every frame at 2 Hz with TTA and temperature scaling, logs to CSV, saves annotated captures. Use this to verify the classifier before wiring in audio.
- **`voice_loop.py`** — voice loop without camera ID. Requires `--species` as a CLI argument; useful for testing the Q&A pipeline against a known species.
- **`run_pipeline.py`** — text-only REPL and latency profiler. No audio hardware needed.

---

## Pipeline

Species identification happens **once at session start**. The camera is opened, one frame is grabbed and TTA-classified, the result is announced via TTS, and the camera is released. All subsequent turns are voice-only — no camera work per question. This keeps per-turn latency clean.

```
                    ── STARTUP ──
                          │
           ┌──────────────┴──────────────┐
           │                             │
    Load Piper TTS               Load Whisper STT
    (speaks errors)              (background thread,
                                  hides load time)
           │                             │
           ▼                             │
  ONE-SHOT SPECIES ID                    │
  Open camera (V4L2)                     │
  Flush 5 stale buffer frames            │
  Grab 1 frame                           │
  TFLiteClassifierTTA (live_inference_mv3.py)
    ├─ 4 TTA views                       │
    ├─ temperature scaling               │
    └─ averaged probabilities            │
  conf ≥ 0.55 & not seashore?            │
  → speak "I see a <species>"            │
  Camera released                        │
           │                             │
           └──────────────┬──────────────┘
                          │  (join whisper thread)
                          ▼
               Build RoboRangerPipeline
                          │
                     warmup query
                          │
                 ══ VOICE LOOP (per turn) ══
                          │
              push-to-talk: Enter to record
                          │
              whisper.transcribe()
                          │
                          ▼
    ┌──────────────────────────────────────────────┐
    │             RoboRangerPipeline               │
    │                                              │
    │  1. WildlifeGate                             │
    │     Two embedding centroids.                 │
    │     Any positive margin → accept.            │
    │     off-topic? → fixed message, loop         │
    │                                              │
    │  2. BlurbStore.get(species_id)               │
    │     SQLite lookup → structured Blurb.        │
    │     unknown? → fixed message, loop           │
    │                                              │
    │  3. IntentClassifier                         │
    │     8 intents. Same bge-small embedder.      │
    │     OTHER / low confidence? → fixed message  │
    │                                              │
    │  4. Direct route?                            │
    │     High-conf simple intent:                 │
    │     format blurb field, skip LLM    ←── sub-ms
    │                                              │
    │  5. sqlite-vec retrieval                     │
    │     Top-K Wikipedia chunks,                  │
    │     filter: cosine ≥ 0.55                    │
    │                                              │
    │  6. build_messages()                         │
    │     Intent-specific prompt template.         │
    │     Polarity-aware danger framing.           │
    │                                              │
    │  7. LlamaCppBackend → llama.cpp :8080        │
    │     SmolLM2-360M-q8_0, max_tokens=100        │
    └──────────────────────────────────────────────┘
                          │
              piper.synthesize_wav() → aplay
                          │
                     back to loop
```

### Stage descriptions

| Stage | File | What it does |
|---|---|---|
| Vision classifier | `live_inference_mv3.py` | MobileNetV3 INT8 TFLite. 4 TTA views averaged, temperature scaled. Called once at startup, not per turn. |
| STT | `full_roboranger_run.py` | pywhispercpp wrapping whisper.cpp. `ggml-tiny.en-q5_1.bin`. Loaded in a background thread during camera ID. |
| WildlifeGate | `species_identification/llm-tuning/wildlife_gate.py` | Embedding cosine sim against wildlife vs. off-topic prototype centroids. Any positive margin → accept. |
| BlurbStore | `species_identification/pipeline/blurb_store.py` | SQLite lookup returning a typed `Blurb` dataclass. Defensive against bad JSON, missing fields, enum typos. |
| IntentClassifier | `species_identification/llm-tuning/intent.py` | Per-intent prototype centroids. Outputs intent + confidence (high/medium/low). Same embedder, no extra model load. |
| Direct route | `species_identification/pipeline/pipeline.py` | High-confidence simple intents (danger, diet, habitat, size, behavior, description) answered from blurb fields — no retrieval, no LLM. Sub-millisecond. |
| Retrieval | `species_identification/tests/test_corpus.py` | sqlite-vec ANN search over Wikipedia chunks. L2 distance → cosine; chunks below threshold dropped. |
| Prompt builder | `species_identification/llm-tuning/prompt_builder.py` | Assembles system + user messages. Intent drives which blurb fields lead and which template is used. Handles danger polarity. |
| LLM | `species_identification/llm-tuning/llm_backends.py` | `LlamaCppBackend` posts to llama.cpp's `/v1/chat/completions` at `localhost:8080`. temp=0.2, max_tokens=100. |
| TTS | `full_roboranger_run.py` | piper-tts synthesizes to a temp WAV, played via `aplay`. |

---

## Build Workflow (laptop — not the UNO Q)

The code runs on Python 3.11, since the newer verions break the package dependencies. The device ships two pre-built artifacts: **`model_int8.tflite`** and **`corpus.db`**. Build these offline and transfer them.

```
1. Train MobileNetV3 classifier
   python species_identification/cnn/mobile_net_v3s.py  (or mobile_net_v3l.py)
       └── Keras model + temperature.json saved to outputs/

2. Export INT8 TFLite
   python species_identification/export_int8_tflite_subprocess.py
       └── model_int8.tflite + class_names.json + temperature.json

3. Build species corpus  (run BEFORE compile_blurbs)
   python species_identification/build_corpus.py species.json corpus.db
       Scrapes Wikipedia, chunks text, embeds with bge-small-en-v1.5,
       stores vectors in corpus.db via sqlite-vec.

4. Generate structured blurbs
   python species_identification/build_blurbs.py   (calls local Ollama)
       Drafts per-field blurbs: diet, habitat, danger, behavior, etc.

5. Compile blurbs into corpus.db
   python species_identification/compile_blurbs.py
       Embeds blurb text and writes blurb_json into the species table.

Transfer to device:  model_int8.tflite  class_names.json  temperature.json  corpus.db
```

Install laptop deps:

```bash
pip install -r requirements-build.txt
```

---

## Device Setup (Arduino UNO Q)

### 1. Install runtime dependencies

```bash
sudo apt install build-essential cmake   # needed by pywhispercpp (builds from source)
pip install --break-system-packages -r requirements.txt
```

### 2. Files required on device

| File | What it is |
|---|---|
| `model_int8.tflite` | INT8 vision classifier (built on laptop, transferred) |
| `class_names.json` | Class labels — must match the TFLite model exactly |
| `temperature.json` | Temperature scaling factor `T` (optional, defaults to 1.0) |
| `corpus.db` | SQLite + sqlite-vec species knowledge base |
| `models/ggml-tiny.en-q5_1.bin` | Whisper STT weights |
| `voices/en_US-lessac-low.onnx` + `.onnx.json` | Piper TTS voice |
| `smollm2-360m-instruct-q8_0.gguf` | LLM weights |

The actual model path on the UNO Q is:
```
/home/arduino/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct-GGUF/
  snapshots/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf
```
A symlink at `~/models/smollm2-360m.gguf` points to it.

### 3. Start the llama.cpp server

```bash
cd ~/llama.cpp/build/bin
./llama-server \
    -m ~/models/smollm2-360m.gguf \
    --host 0.0.0.0 --port 8080 \
    --ctx-size 4096 --threads 4 \
    --chat-template chatml \
    --cache-prompt
```

`~/models/smollm2-360m.gguf ` is a symbolic link to the actual location for the installed model, so replace it with the actual path to the model.
`--cache-prompt` is important: it caches the system prompt + species blurb prefix so subsequent queries on the same species are faster.

---

## Running

### Full integrated loop (primary entry point)

```bash
python3 full_roboranger_run.py \
    --voice voices/en_US-lessac-low.onnx \
    --whisper-model models/ggml-tiny.en-q5_1.bin \
    --backend llama-cpp
```

Identifies species from the camera at startup, speaks the result, then enters push-to-talk Q&A. Press Enter to start recording, Enter again to stop.

**Override species** (skip the camera — for testing on a laptop):
```bash
python3 full_roboranger_run.py \
    --species Zalophus_californianus \
    --voice voices/en_US-lessac-low.onnx \
    --whisper-model models/ggml-tiny.en-q5_1.bin \
    --backend llama-cpp
```

**Verify the camera without loading any models:**
```bash
python3 full_roboranger_run.py --camera-test
```

### Camera-only inference (no voice)

```bash
python3 live_inference_mv3.py
```

Reads from the camera at 2 Hz. Annotated frames saved to `species_identification/outputs/captures_tta/`. Predictions logged to `species_identification/outputs/predictions_tta.csv`. Use this to validate the classifier before wiring in audio.

### Text REPL (no audio hardware)

```bash
# Interactive
python3 run_pipeline.py --species Zalophus_californianus --repl --backend llama-cpp

# One-shot
python3 run_pipeline.py --species Zalophus_californianus --query "is it dangerous?" --backend llama-cpp

# Per-stage latency profile over a query file
python3 run_pipeline.py --species Zalophus_californianus \
    --profile species_identification/tests/test_queries.txt --backend llama-cpp
```

Use `--backend ollama` (default) for laptop iteration against a local Ollama server.

---

## File Layout

```
.
├── full_roboranger_run.py                # ← primary entry point: camera ID + voice loop
├── live_inference_mv3.py                 # standalone camera inference loop (MobileNetV3 TTA)
├── run_pipeline.py                       # text REPL / one-shot / latency profiler
├── voice_loop.py                         # voice loop without camera ID (--species required)
├── requirements.txt                      # on-device runtime deps (pinned)
├── requirements-build.txt                # laptop-only build deps
│
└── species_identification/
    ├── pipeline/
    │   ├── pipeline.py                   # orchestrator (gate → blurb → intent → route → LLM)
    │   ├── pipeline_factory.py           # builds a fully wired pipeline from config
    │   ├── pipeline_repl.py              # interactive REPL for the pipeline alone
    │   └── blurb_store.py               # structured blurb lookup from corpus.db
    │
    ├── llm-tuning/
    │   ├── wildlife_gate.py             # on/off-topic binary gate (embedding centroids)
    │   ├── intent.py                    # 8-way intent classifier (embedding centroids)
    │   ├── llm_backends.py              # OllamaBackend + LlamaCppBackend (stdlib only)
    │   └── prompt_builder.py            # prompt assembly; danger polarity handling
    │
    ├── tts-and-stt/
    │   ├── ptt_whisper.py               # push-to-talk STT prototype
    │   ├── ptt_test.py                  # audio hardware verification
    │   ├── audition.py                  # Piper TTS test harness
    │   ├── whisper_test.py              # Whisper accuracy/latency test
    │   └── verify_mic.py               # mic input check
    │
    ├── cnn/
    │   ├── mobile_net_v3s.py            # MobileNetV3-Small training (two-phase, TF/Keras)
    │   ├── mobile_net_v3l.py            # MobileNetV3-Large training
    │   ├── mobile_net_v2.py             # MobileNetV2 (earlier iteration)
    │   ├── model_diagnostics.py         # post-training diagnostics
    │   └── web-scraper.py              # additional image collection
    │
    ├── build_corpus.py                  # (laptop) Wikipedia → chunked embeddings → corpus.db
    ├── build_blurbs.py                  # (laptop) Ollama → structured blurbs per species
    ├── compile_blurbs.py                # (laptop) write blurb_json into corpus.db
    ├── generate_prose.py                # (laptop) prose refinement pass over blurbs
    ├── export_tflite.py                 # (laptop) float32 TFLite export
    ├── export_int8_tflite_subprocess.py # (laptop) INT8 TFLite via onnx2tf
    │
    └── tests/
        ├── test_corpus.py               # corpus retrieval utilities + open_db()
        ├── test_tflite.py               # TFLite interpreter smoke test
        ├── test_pipeline_integration.py
        ├── test_intent_units.py         # intent classifier unit tests
        ├── test_phrasing.py             # prompt phrasing regression tests
        ├── eval_pipeline.py             # pipeline eval (path distribution, accuracy)
        ├── eval_harness.py              # retrieval threshold tuning
        ├── eval_e2e.py                  # end-to-end answer quality eval
        ├── eval_intent.py               # intent classifier eval
        ├── diff_runs.py                 # compare two eval result sets
        ├── verify_matches_db.py         # corpus coverage per species
        ├── preview_danger_prompts.py    # inspect built danger prompts
        └── mem_check.py                 # RSS memory tracking (called in voice loop)
```

---

## Key Tuning Constants

| Constant | Location | Default | Effect |
|---|---|---|---|
| `ID_CONF_THRESHOLD` | `full_roboranger_run.py` | 0.55 | Minimum classifier confidence to accept a species identification at startup |
| `CAMERA_WARMUP_FRAMES` | `full_roboranger_run.py` | 5 | Frames flushed before capture to clear stale V4L2 buffer |
| `CONF_THRESH` | `live_inference_mv3.py` | 0.15 | Display threshold for the standalone camera loop (below = grey text) |
| `INFER_HZ` | `live_inference_mv3.py` | 2.0 | Camera inference rate in the standalone loop. TTA runs 4 passes per tick. |
| `DEFAULT_RETRIEVAL_THRESHOLD` | `pipeline.py` | 0.55 | Cosine similarity floor for RAG chunks |
| `DIRECT_ROUTE_SCORE_THRESHOLD` | `pipeline.py` | 0.65 | Minimum intent score for medium-confidence direct routing (skips LLM) |
| `WildlifeGate.ACCEPT_MARGIN` | `wildlife_gate.py` | 0.05 | Wildlife–off-topic margin above which gate accepts with high confidence |
| `SAMPLING["max_tokens"]` | `llm_backends.py` | 100 | LLM output cap (tuned for UNO Q speed; 200 was too slow) |
| `EMBED_MODEL` | `test_corpus.py` | `BAAI/bge-small-en-v1.5` | Must match the model used at corpus build time |
