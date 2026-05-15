# RoboRanger — On-Device Wildlife Guide

A portable AI wildlife guide that runs **fully locally** on the **Arduino UNO Q** (quad-core SoC, Linux Debian). A visitor points the camera at an animal or plant, asks a question out loud, and gets a spoken answer — no internet connection required after setup.

Scoped to **La Jolla Cove** species, but the pipeline is locale-general: swap the classifier and corpus for any location.

Devpost: https://devpost.com/software/roboranger?ref_content=my-projects-tab&ref_feature=my_projects

---

## How It Works

Three entry points share the same pipeline core:

- **`live_inference.py`** — headless camera loop. Classifies every frame at 2 Hz using Test-Time Augmentation (4 views per frame) and temperature scaling. Logs predictions and saves annotated captures. Use this to verify the classifier before wiring in voice.
- **`voice_loop.py`** — the full push-to-talk loop. Records a question, transcribes it with Whisper, runs the Q&A pipeline, and speaks the answer back with Piper TTS.
- **`run_pipeline.py`** — text-only REPL and latency profiler. Useful for testing the pipeline without audio hardware.

---

## Pipeline

```
Camera frame
    │
    ▼
MobileNetV3-Small/Large (INT8 TFLite)        ← live_inference.py
    │  top-1 species_id + confidence
    │
    ▼
Microphone (push-to-talk, Enter key on device)
    │
    ▼
Whisper.cpp  (ggml-tiny.en-q5_1.bin)         ← STT
    │  query text
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  RoboRangerPipeline                  │
│                                                      │
│  1. WildlifeGate                                     │
│     Two embedding centroids (wildlife vs. off-topic).│
│     Any positive margin → accept. Rejects bathroom,  │
│     parking, chitchat before touching the DB.        │
│                     │                                │
│                     ▼                                │
│  2. BlurbStore.get(species_id)                       │
│     SQLite lookup → structured Blurb:                │
│     appearance, diet, habitat, size, behavior,       │
│     dangerous_to_humans, dangerous_to_pets, etc.     │
│     Returns None if species isn't in the corpus.     │
│                     │                                │
│                     ▼                                │
│  3. IntentClassifier                                 │
│     Classifies query into one of 8 intents:          │
│     DESCRIPTION, DANGER, DIET, HABITAT, SIZE,        │
│     BEHAVIOR, IDENTIFICATION, CONSERVATION.          │
│     Uses the same bge-small embedder — no extra load.│
│                     │                                │
│          ┌──────────┴──────────┐                     │
│          │                     │                     │
│     High-confidence        Everything else           │
│     simple intent               │                    │
│          │                      ▼                    │
│          ▼               4. Retrieval                │
│    Direct answer          sqlite-vec ANN search over │
│    from blurb field       Wikipedia chunks embedded  │
│    (sub-millisecond,      at build time. Chunks below│
│    no LLM needed)         cosine ≥ 0.55 are dropped. │
│                                 │                    │
│                                 ▼                    │
│                         5. Prompt assembly           │
│                                 │                    │
│                                 ▼                    │
│                         6. LlamaCppBackend           │
│                            smollm2-360m-instruct     │
│                            via llama.cpp at :8080    │
└──────────────────────────────────────────────────────┘
    │  answer text
    ▼
Piper TTS  (en_US-lessac-low.onnx)           ← TTS
    │
    ▼
Speaker (aplay)
```

### Stage descriptions

| Stage | File | What it does |
|---|---|---|
| Vision classifier | `live_inference.py` | MobileNetV3 INT8 TFLite. 4 TTA views averaged, temperature scaled. Runs at 2 Hz. |
| STT | `voice_loop.py` | pywhispercpp wrapping whisper.cpp. `ggml-tiny.en-q5_1.bin` for speed. |
| WildlifeGate | `species_identification/llm-tuning/wildlife_gate.py` | Embedding cosine sim against wildlife vs. off-topic prototype centroids. Any positive margin → accept. |
| BlurbStore | `species_identification/pipeline/blurb_store.py` | SQLite lookup returning a typed `Blurb` dataclass. Defensive against bad JSON, missing fields, enum typos. |
| IntentClassifier | `species_identification/llm-tuning/intent.py` | Per-intent prototype centroids. Outputs intent + confidence (high/medium/low). Same embedder, no extra model load. |
| Direct route | `species_identification/pipeline/pipeline.py` | High-confidence simple intents (danger, diet, habitat, size, behavior, description) are answered from blurb fields directly — no retrieval, no LLM. Sub-millisecond. |
| Retrieval | `species_identification/tests/test_corpus.py` | sqlite-vec ANN search over Wikipedia chunks. L2 distance → cosine; chunks below threshold dropped. |
| Prompt builder | `species_identification/llm-tuning/prompt_builder.py` | Assembles system + user messages. Intent drives which blurb fields lead and which template is used. |
| LLM | `species_identification/llm-tuning/llm_backends.py` | `LlamaCppBackend` posts to llama.cpp's `/v1/chat/completions` at `localhost:8080`. temp=0.2, max_tokens=100. |
| TTS | `voice_loop.py` | piper-tts synthesizes to a temp WAV, played via `aplay`. |

---

## Build Workflow (laptop — not the UNO Q)

The device ships two pre-built artifacts: **`model_int8.tflite`** and **`corpus.db`**. Build these offline and transfer them.

```
1. Train MobileNetV3 classifier
   species_identification/cnn/mobile_net_v3s.py  (or mobile_net_v3l.py)
       └── PyTorch checkpoint → ONNX

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

`--cache-prompt` is important: it caches the system prompt + species blurb prefix so subsequent queries on the same species are faster.

---

## Running

### Camera-only inference (no voice)

```bash
python3 live_inference.py
```

Reads from `/dev/video0`. Annotated frames saved to `captures_tta/`. Predictions logged to `predictions_tta.csv`.

### Full voice loop

```bash
python3 voice_loop.py \
    --species Zalophus_californianus \
    --voice voices/en_US-lessac-low.onnx \
    --whisper-model models/ggml-tiny.en-q5_1.bin \
    --backend llama-cpp
```

Press Enter to start recording, press Enter again to stop. The pipeline answers and Piper speaks the response.

### Text REPL (no audio hardware)

```bash
# Interactive
python3 run_pipeline.py --species Zalophus_californianus --repl --backend llama-cpp

# One-shot
python3 run_pipeline.py --species Zalophus_californianus --query "is it dangerous?" --backend llama-cpp

# Latency profile over a query file
python3 run_pipeline.py --species Zalophus_californianus \
    --profile species_identification/tests/test_queries.txt --backend llama-cpp
```

Use `--backend ollama` (default) for laptop iteration against a local Ollama server.

---

## File Layout

```
.
├── run_pipeline.py                       # text REPL / one-shot / latency profiler
├── voice_loop.py                         # full push-to-talk voice loop
├── live_inference.py                     # headless camera inference (TTA + temperature)
├── requirements.txt                      # on-device runtime deps
├── requirements-build.txt                # laptop-only build deps
│
└── species_identification/
    ├── pipeline/
    │   ├── pipeline.py                   # orchestrator (gate → blurb → intent → retrieve → LLM)
    │   ├── pipeline_factory.py           # builds a fully wired pipeline from CLI args
    │   ├── pipeline_repl.py              # thin REPL wrapper
    │   └── blurb_store.py               # reads structured blurbs from corpus.db
    │
    ├── llm-tuning/
    │   ├── wildlife_gate.py             # on/off-topic binary classifier (embedding)
    │   ├── intent.py                    # 8-way intent classifier (embedding)
    │   ├── llm_backends.py              # OllamaBackend + LlamaCppBackend (stdlib only)
    │   └── prompt_builder.py            # assembles chat messages from blurb + chunks + intent
    │
    ├── tts-and-stt/
    │   ├── ptt_whisper.py               # push-to-talk STT prototype
    │   └── audition.py                  # Piper TTS test harness
    │
    ├── cnn/
    │   ├── mobile_net_v3s.py            # MobileNetV3-Small training script
    │   ├── mobile_net_v3l.py            # MobileNetV3-Large training script
    │   └── web-scraper.py               # additional image scraping if needed
    │
    ├── build_corpus.py                  # (laptop) fetch Wikipedia, chunk, embed → corpus.db
    ├── build_blurbs.py                  # (laptop) generate structured blurbs via Ollama
    ├── compile_blurbs.py                # (laptop) embed + write blurbs into corpus.db
    ├── generate_prose.py                # (laptop) prose refinement pass over blurbs
    ├── export_tflite.py                 # (laptop) float32 TFLite export
    ├── export_int8_tflite_subprocess.py # (laptop) INT8 TFLite via onnx2tf
    │
    └── tests/
        ├── test_tflite.py               # TFLite interpreter smoke test
        ├── test_corpus.py               # corpus retrieval utilities (open_db, retrieve)
        ├── test_pipeline_integration.py
        ├── eval_pipeline.py             # pipeline-level eval
        ├── eval_harness.py              # retrieval threshold tuning harness
        └── eval_e2e.py                  # end-to-end eval
```

---

## Key Tuning Constants

| Constant | Location | Default | Effect |
|---|---|---|---|
| `CONF_THRESH` | `live_inference.py` | 0.15 | Predictions below this are shown grey; adjust for your classifier's calibration |
| `INFER_HZ` | `live_inference.py` | 2.0 | Camera inference rate. TTA runs 4 passes per tick. |
| `DEFAULT_RETRIEVAL_THRESHOLD` | `pipeline.py` | 0.55 | Cosine similarity floor for RAG chunks |
| `DIRECT_ROUTE_SCORE_THRESHOLD` | `pipeline.py` | 0.65 | Minimum intent score for medium-confidence direct routing (skips LLM) |
| `WildlifeGate.ACCEPT_MARGIN` | `wildlife_gate.py` | 0.05 | Margin above which gate accepts with high confidence |
| `SAMPLING["max_tokens"]` | `llm_backends.py` | 100 | LLM output cap (tuned for UNO Q speed; 200 was too slow) |
| `EMBED_MODEL` | `test_corpus.py` | `BAAI/bge-small-en-v1.5` | Must match the model used at corpus build time |
