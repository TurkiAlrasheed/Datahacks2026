# RoboRanger — On-Device Wildlife Guide

A portable AI wildlife guide that runs **fully locally** on the **Arduino UNO Q** (quad-core SoC, Linux Debian). A visitor points the camera at an animal or plant, asks a question out loud, and gets a spoken answer — no internet connection required after setup.

The on-device classifier and corpus cover **50 species from the UCSD / San Diego iNaturalist set** (plus a `seashore` "nothing to narrate" class). The pipeline is locale-general: swap the classifier and corpus for any location.

Devpost: https://devpost.com/software/roboranger?ref_content=my-projects-tab&ref_feature=my_projects

---

## How It Works

The primary entry point is **`full_roboranger_run.py`**, which integrates all four components into one session. On startup it identifies the species from the camera once, announces what it sees, then enters a push-to-talk Q&A loop for the rest of the session. Utility scripts cover isolated testing:

- **`full_roboranger_run.py`** ← main entry point. Loads Piper first (so it can speak errors), checks the corpus, loads Whisper in a background thread while camera ID runs, classifies one frame, announces the species (or says why it couldn't and lets you retry), then loops on push-to-talk questions.
- **`live_inference_mv3.py`** — headless camera loop only, for any model directory (`--model-dir`). Classifies every frame at 2 Hz with TTA and temperature scaling, logs to a per-model CSV, saves annotated captures. `live_inference_mv2.py` is the same loop defaulting to the MobileNetV2-QAT model.
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
    Check corpus.db schema        hides load time)
           │                             │
           ▼                             │
  ONE-SHOT SPECIES ID                    │
  Open camera (V4L2)                     │
  Flush 5 stale buffer frames            │
  Grab 1 frame                           │
  TFLiteClassifierTTA                    │
  (vision/tflite_classifier.py)          │
    ├─ preprocessing from manifest       │
    ├─ 4 TTA views                       │
    ├─ temperature scaling               │
    └─ averaged probabilities            │
  conf ≥ id_threshold & not seashore?    │
  → speak "I see a <species>"            │
    (else say why and retry)             │
  Camera released                        │
           │                             │
           └──────────────┬──────────────┘
                          │  (join whisper thread)
                          ▼
               Build RoboRangerPipeline
                          │
       warmup: retrieval + one LLM call for the species
       write device_status.json (corpus + model versions)
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
    │     (missing field → falls through)          │
    │                                              │
    │  5. sqlite-vec retrieval                     │
    │     Exact top-K in the species' partition,   │
    │     blurb excluded, cosine ≥ 0.55            │
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
| Vision classifier | `species_identification/vision/tflite_classifier.py` | Manifest-driven TFLite classifier. Preprocessing (input domain, size, dtype), class order, temperature and threshold all come from `model_manifest.json`. 4 TTA views averaged, temperature scaled. Called once at startup, not per turn. |
| STT | `full_roboranger_run.py` | pywhispercpp wrapping whisper.cpp. `ggml-tiny.en-q5_1.bin`. Loaded in a background thread during camera ID. |
| WildlifeGate | `species_identification/llm-tuning/wildlife_gate.py` | Embedding cosine sim against wildlife vs. off-topic prototype centroids. Any positive margin → accept. |
| BlurbStore | `species_identification/pipeline/blurb_store.py` | SQLite lookup returning a typed `Blurb` dataclass. Defensive against bad JSON, missing fields, enum typos. |
| IntentClassifier | `species_identification/llm-tuning/intent.py` | Per-intent prototype centroids. Outputs intent + confidence (high/medium/low). Same embedder, no extra model load. |
| Direct route | `species_identification/pipeline/pipeline.py` | High-confidence simple intents (danger, diet, habitat, size, behavior, description) answered from blurb fields — no retrieval, no LLM. Sub-millisecond. A missing field falls through to retrieval + LLM. |
| Retrieval | `species_identification/tests/test_corpus.py` | Exact KNN inside the species' sqlite-vec partition (`species_id` partition key), excluding the blurb chunk. L2 distance → cosine; chunks below threshold dropped. |
| Prompt builder | `species_identification/llm-tuning/prompt_builder.py` | Assembles system + user messages. Intent drives which blurb fields lead and which template is used. Handles danger polarity. |
| LLM | `species_identification/llm-tuning/llm_backends.py` | `LlamaCppBackend` posts to llama.cpp's `/v1/chat/completions` at `localhost:8080`. temp=0.2, max_tokens=100 (same for the Ollama backend used on the laptop). |
| TTS | `full_roboranger_run.py` | piper-tts synthesizes to a temp WAV, played via `aplay`. |

---

## Build Workflow (laptop — not the UNO Q)

The code runs on Python 3.11, since newer versions break the package dependencies. The device ships two pre-built artifacts: a **model directory** (`model_*.tflite` + `model_manifest.json`) and **`corpus.db`**. Build these offline and transfer them. All commands run from the repo root.

```
1. Train the classifier (writes to species_identification/outputs/<arch>/)
   python species_identification/cnn/mobile_net_v3l.py      (or mobile_net_v3s.py / mobile_net_v2.py)
   Optional float-model diagnostics (per-class metrics, confusion matrix, most-confused pairs):
   python species_identification/cnn/model_diagnostics.py --arch mobilenetv3l
       Reads final.keras, class_names.json and temperature.json from outputs/<arch>/
       (--model/--classes/--temp/--splits override), writes outputs/<arch>/diagnostics/.
       Scales input the way that arch was trained (raw 0..255 for MobileNetV3,
       x/127.5 - 1 for mobilenetv2_qat) and exits if the model's graph contradicts --arch.

2. Export to TFLite + manifest (Keras models)
   python species_identification/export_tflite.py \
       --keras species_identification/outputs/mobilenetv3l/best.keras \
       --arch mobilenetv3l --quant dynamic \
       --out species_identification/outputs/mobilenetv3l_dynamic
       └── model_dynamic.tflite + model_manifest.json
   (PyTorch/DINOv2 models use export_int8_tflite_subprocess.py instead — ImageNet mean/std input.)

3. Evaluate on the device path (LiteRT + 4-view TTA), fit temperature + threshold into the manifest
   python species_identification/tests/eval_tflite.py \
       --model-dir species_identification/outputs/mobilenetv3l_dynamic \
       --keras species_identification/outputs/mobilenetv3l/best.keras --workers 5 --write
   --keras (optional) also reports top-1 agreement between the TFLite and float Keras models.
   Then copy the chosen model dir to species_identification/outputs/deploy/.

4. Build species corpus  (run BEFORE compile_blurbs)
   python species_identification/build_corpus.py species.json corpus.db
       Scrapes Wikipedia, chunks text, embeds with bge-small-en-v1.5,
       stores vectors in a species-partitioned sqlite-vec table (schema v2).

5. Generate structured blurbs
   python species_identification/build_blurbs.py species.json blurbs.yaml   (calls local Ollama)
       Drafts per-field blurbs: diet, habitat, danger, behavior, etc.
       Review by hand and set reviewed: true.

6. Compile blurbs into corpus.db
   python species_identification/compile_blurbs.py blurbs.yaml corpus.db
       Writes blurb_json into the species table and refreshes the corpus manifest.

Existing schema v1 corpus?  (global vector index, no species partition)
   python species_identification/migrate_corpus_v2.py old_corpus.db corpus_v2.db
       Copies rows and vectors byte-for-byte (no re-scrape, no re-embed), verifies, writes v2.

Transfer to device:  species_identification/outputs/deploy/  and  corpus.db
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
| `species_identification/outputs/deploy/model_*.tflite` | Vision classifier (built on laptop, transferred) |
| `species_identification/outputs/deploy/model_manifest.json` | Input preprocessing, class order, temperature, identification threshold — must travel with the model |
| `species_identification/offline-info/corpus.db` | SQLite + sqlite-vec species knowledge base, schema v2 (needs sqlite-vec ≥ 0.1.6; device pins 0.1.9) |
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

`~/models/smollm2-360m.gguf` is a symbolic link to the actual location for the installed model, so replace it with the actual path to the model.
`--cache-prompt` reuses the longest common prompt prefix between requests. The system prompt and the species header are identical across turns; the order of the blurb fields depends on the question's intent, so only turns with the same intent reuse the whole SPECIES FACTS block. The startup warmup primes the cache for the identified species.

---

## Running

### Full integrated loop (primary entry point)

```bash
python3 full_roboranger_run.py \
    --voice voices/en_US-lessac-low.onnx \
    --whisper-model models/ggml-tiny.en-q5_1.bin \
    --backend llama-cpp
```

Identifies species from the camera at startup, speaks the result, then enters push-to-talk Q&A. Press Enter to start recording, Enter again to stop. If identification fails (not confident, seashore, camera error) it says why and waits for Enter to try again. `--model-dir` picks the classifier (default `species_identification/outputs/deploy`); `device_status.json` records the corpus and model versions for the session.

**Override species** (skip the camera — for testing on a laptop):
```bash
python3 full_roboranger_run.py \
    --species Canis_latrans \
    --voice voices/en_US-lessac-low.onnx \
    --whisper-model models/ggml-tiny.en-q5_1.bin \
    --backend llama-cpp
```

**Verify the camera without loading any models:**
```bash
python3 full_roboranger_run.py --camera-test
```

Saves one frame to `camera_test_<timestamp>.jpg` and exits. No voice, Whisper, or classifier files are needed; `--voice` and `--whisper-model` are only required for the full loop.

### Camera-only inference (no voice)

```bash
python3 live_inference_mv3.py --model-dir species_identification/outputs/deploy --camera-index 0
```

Reads from the camera at 2 Hz. Annotated frames and `predictions_tta.csv` go to `species_identification/outputs/live/<model dir name>/`, so different models never share a log. Use this to validate the classifier (and its on-device speed) before wiring in audio.

### Text REPL (no audio hardware)

```bash
# Interactive
python3 run_pipeline.py --species Canis_latrans --repl --backend llama-cpp

# One-shot
python3 run_pipeline.py --species Canis_latrans --query "is it dangerous?" --backend llama-cpp

# Per-stage latency profile over a query file
python3 run_pipeline.py --species Canis_latrans \
    --profile test_queries.txt --backend llama-cpp
```

Use `--backend ollama` (default) for laptop iteration against a local Ollama server.

---

## Retrieval: species-partitioned vector index

Retrieval always knows the `species_id` before it searches, so the search is scoped to that species inside sqlite-vec:

```sql
CREATE VIRTUAL TABLE chunk_vectors USING vec0(
    species_id TEXT PARTITION KEY,
    category   TEXT,
    embedding  float[384],
    chunk_size = 16
);

WITH knn AS (
    SELECT rowid, distance FROM chunk_vectors
    WHERE embedding MATCH ? AND k = ? AND species_id = ? AND category != 'blurb'
)
SELECT c.id, c.category, c.text, knn.distance FROM knn JOIN chunks c ON c.id = knn.rowid;
```

**Why.** Schema v1 had a single global vec0 index. The query took the 50 globally nearest chunks and only then filtered to the species, so any species whose chunks weren't in that global top 50 silently got fewer (or zero) snippets. It also returned the species' own blurb chunk, which the prompt already contains as SPECIES FACTS. `vec0` is an exact brute-force index (not ANN) in both schemas; the partition key scopes the scan to one species.

**Measured** with `species_identification/tests/retrieval_recall_check.py` (50 species, 871 chunks, 8 generic visitor questions per species = 400 queries). Ground truth is an exact `vec_distance_l2` scan over the species' non-blurb chunks.

| | global k=50 + post-filter (v1) | species partition (v2) |
|---|---|---|
| queries missing a better chunk, with name grounding | 20.2% | **0%** |
| queries returning **zero** chunks, without name grounding | 52.5% | **0%** |
| prompts carrying a duplicate blurb snippet | 74.5% | **0%** |
| retrieval latency p50 (laptop) | 2.38 ms | **0.21 ms** |
| `corpus.db` size | 2.87 MB | 3.37 MB |

Raising the global `k` only shrinks the v1 loss slowly: at k=500, 6% of queries were still missing chunks (not counting the 3 species that have fewer than 5 chunks in total), and it only reaches zero once `k` covers the entire corpus — a number that grows with every species added. vec0 allocates one vector chunk per partition, and the default `chunk_size=1024` would have made this corpus **81 MB**; `chunk_size=16` keeps it at 3.37 MB (measured with `migrate_corpus_v2.py --measure`).

The runtime refuses a v1 corpus at startup (`check_corpus_schema`) with a pointer to `migrate_corpus_v2.py`, rather than silently falling back to the global search.

---

## Vision model: preprocessing contract and quantization

Each exported model expects a different input domain, and nothing in a `.tflite` says which:

| Model | Input the graph expects | I/O dtype |
|---|---|---|
| MobileNetV3 (Keras, `include_preprocessing=True`) | raw pixels 0..255 (rescale is inside the graph) | uint8 (int8 export) / float32 |
| MobileNetV2 QAT (Keras) | `x / 127.5 - 1` | int8 |
| DINOv2 via ONNX → onnx2tf | ImageNet mean/std | int8 |

Feeding the wrong domain doesn't crash — it returns confident garbage. So every model directory carries a `model_manifest.json` (`input_range`, `img_size`, `input_dtype`, `class_names`, `temperature`, `id_threshold`, TTA views, provenance and eval summary), and `TFLiteClassifierTTA` refuses to run without one or when the model's tensors disagree with it.

Two problems this uncovered:

1. **Training runs overwrote each other.** All three trainers wrote `best.keras`, `temperature.json`, `class_names.json`, `test_results.json` and `model_fp32.tflite` into the same `outputs/`. The deployed MobileNetV3-Large int8 file dated from May 19, while `temperature.json` came from a May 22 retrain that was never exported; `model_fp32.tflite` was actually the MobileNetV2 model; and both live-inference scripts shared one temperature file and one predictions log. Trainers now write to `outputs/<arch>/`, and temperature/threshold live in each model's manifest. `cnn/model_diagnostics.py` had the same problem: it read `best.keras` and `temperature.json` from the shared folder and assumed 320 px input for every model. It now takes `--arch`, reads from `outputs/<arch>/`, gets the image size from the model, and applies that architecture's input scaling.
2. **Full-integer post-training quantization breaks MobileNetV3-Large.** Its hard-swish / squeeze-excite activations don't survive int8 PTQ. Measured through the device path (LiteRT, 4-view TTA) on a 2-images-per-class slice of test:

| Model | TTA top-1 (slice) | agrees with float model |
|---|---|---|
| MobileNetV3-L full-int8 (deployed, May 19) | 22.5% | — |
| MobileNetV3-L full-int8 (re-exported) | 16.7% | 13.7% (1-per-class) |
| MobileNetV2 QAT int8 | 64.7% | — |
| MobileNetV3-L dynamic-range (int8 weights, float activations) | 70.6% | 92.2% |

The preprocessing in `live_inference_mv3.py` was actually correct for the MobileNetV3 graph; MobileNetV2 looked better because it was quantization-aware trained, not because of normalization.

The two working candidates on the **full** test split (777 images), each with temperature and threshold fit on the full val split:

| Model | TTA top-1 | Top-5 | ECE | Threshold | Precision | Coverage | Seashore announced |
|---|---|---|---|---|---|---|---|
| **MobileNetV3-L dynamic-range** (shipped) | **79.5%** | **94.2%** | **2.3%** | 0.63 | **91.1%** | 59.6% | 0% |
| MobileNetV2 QAT int8 (fallback) | 78.8% | 92.8% | 2.6% | 0.60 | 87.6% | 60.7% | 0% |

They're close. MobileNetV3-L dynamic-range is slightly more accurate and the only one that holds the 90% precision target on test; MobileNetV2-QAT is fully int8 and likely faster on the UNO Q's CPU.

**Shipped model** (`outputs/deploy/`): MobileNetV3-Large, dynamic-range quantized, 3.4 MB. Full test split (777 images), device path with 4-view TTA:

| Metric | Value |
|---|---|
| Top-1 (single view / TTA) | 77.7% / **79.5%** |
| Top-5 (TTA) | 94.2% |
| Top-1 agreement with the float Keras model | 94.1% (float model: 78.9% single-view) |
| Expected calibration error (T = 1.00, fit on val) | 2.3% |
| Identification threshold (lowest with ≥ 90% val precision) | 0.63 |
| Announced species correct (precision) at 0.63 | 91.1% |
| Species images announced correctly (coverage) | 59.6% |
| Seashore frames announced as a species | 0% |
| Reference-kernel (UNO Q resolver) vs default-kernel top-1 | 8 / 8 identical |

With this model, the previous fixed 0.55 threshold gives 87.7% precision / 63.5% coverage on the same split. The numbers and fit are stored in `outputs/deploy/model_manifest.json` under `eval`. Dynamic-range models run float activations, so **benchmark on the UNO Q** (`live_inference_mv3.py` prints Hz) before relying on it; the MobileNetV2-QAT int8 model in `outputs/mobilenetv2_qat/` is the calibrated fallback if it's too slow. `tests/eval_tflite.py --input-range` deliberately forces a wrong input domain: feeding this model `x/127.5 - 1` instead of raw pixels drops it from 70.6% to 2.0% top-1 on the same test slice — chance level for 51 classes, with no error raised.

---

## OTA corpus updates (design)

**Built today (foundations):**

- `corpus_meta` table: `schema_version`, `corpus_version` (build date + content hash), `content_sha256`, `embed_model` / `embed_dim`, `sqlite_vec_version`, `built_at`, species and chunk counts. Refreshed by `build_corpus.py`, `compile_blurbs.py` and `migrate_corpus_v2.py`.
- `species_manifest` table: one sha256 per species over its species row (blurb) and its chunks' category, text and vector bytes. Independent of chunk ids, so identical content rebuilt elsewhere hashes identically.
- At startup the device checks the schema, logs the corpus version and hash, warns about classifier classes with no blurb, and writes `device_status.json` (corpus version + hash, model tflite sha256, temperature, threshold). That file is what a unit reports.

**Design (not implemented):**

1. **Region subsets.** A region release is a pair: a model directory and a corpus whose species set matches the model's non-negative classes (the startup coverage check enforces this). Releases are named `region/corpus_version`.
2. **Deltas.** The server diffs the unit's `species_manifest` against the target release: added, changed and removed species. A delta pack holds, per changed species, the species row plus its chunks and vectors, the target `species_manifest`, and the expected `content_sha256`. At ~17 chunks × (1.5 KB vector + ~1 KB text) that's roughly 45 KB per species: 40 changed species ≈ 2 MB, independent of how large the rest of the corpus is — a 57 MB regional corpus with 40 changed species still ships ~2 MB. A change of `embed_model` invalidates every vector and forces a full corpus.
3. **Apply.** Copy `corpus.db` → `corpus.new.db`. Per species: `DELETE FROM chunk_vectors WHERE species_id = ?`, delete its chunks and species row, insert the new ones — with the partition key this only touches that species' vector chunks. Then `refresh_manifest`, require `content_sha256` to equal the expected value, run `check_corpus_schema` plus one KNN per changed species, and swap.
4. **Rollback.** Two slots: `corpus.db` (active) and `corpus.prev.db` (last known good). The swap is an atomic `os.replace` after verification; if startup checks fail or the loop crashes repeatedly, the device restores `corpus.prev.db`. Models ship the same way, and because temperature and threshold live in the model's manifest, a model update can't leave a stale calibration behind.
5. **Version reporting.** Units upload `device_status.json` when online; the server plans the next delta from the reported `content_sha256`, not from the build date.

---

## File Layout

```
.
├── full_roboranger_run.py                # ← primary entry point: camera ID + voice loop
├── live_inference_mv3.py                 # standalone camera loop, any model dir (--model-dir)
├── live_inference_mv2.py                 # same loop, defaults to the MobileNetV2-QAT model
├── run_pipeline.py                       # text REPL / one-shot / latency profiler
├── voice_loop.py                         # voice loop without camera ID (--species required)
├── requirements.txt                      # on-device runtime deps (pinned)
├── requirements-build.txt                # laptop-only build deps
│
└── species_identification/
    ├── corpus_schema.py                  # corpus.db schema v2, manifest/versioning, schema check
    ├── migrate_corpus_v2.py              # (laptop) v1 corpus → v2, verified, no re-embedding
    ├── vision/
    │   └── tflite_classifier.py          # manifest-driven TFLite classifier with TTA
    │
    ├── pipeline/
    │   ├── pipeline.py                   # orchestrator (gate → blurb → intent → route → LLM) + warmup
    │   ├── pipeline_factory.py           # builds a fully wired pipeline from config
    │   ├── pipeline_repl.py              # interactive REPL for the pipeline alone
    │   └── blurb_store.py                # structured blurb lookup from corpus.db
    │
    ├── llm-tuning/
    │   ├── wildlife_gate.py              # on/off-topic binary gate (embedding centroids)
    │   ├── intent.py                     # 8-way intent classifier (embedding centroids)
    │   ├── llm_backends.py               # OllamaBackend + LlamaCppBackend (stdlib only)
    │   └── prompt_builder.py             # prompt assembly; danger polarity handling
    │
    ├── tts-and-stt/
    │   ├── ptt_whisper.py                # push-to-talk STT prototype
    │   ├── ptt_test.py                   # audio hardware verification
    │   ├── audition.py                   # Piper TTS test harness
    │   ├── whisper_test.py               # Whisper accuracy/latency test
    │   └── verify_mic.py                 # mic input check
    │
    ├── cnn/
    │   ├── mobile_net_v3s.py             # MobileNetV3-Small training → outputs/mobilenetv3s/
    │   ├── mobile_net_v3l.py             # MobileNetV3-Large training → outputs/mobilenetv3l/
    │   ├── mobile_net_v2.py              # MobileNetV2 + QAT training → outputs/mobilenetv2_qat/
    │   ├── model_diagnostics.py          # float-model test diagnostics (--arch) → outputs/<arch>/diagnostics/
    │   └── web-scraper.py                # additional image collection
    │
    ├── outputs/
    │   ├── deploy/                       # the model the device runs (tflite + manifest)
    │   └── <arch>/                       # per-architecture training outputs + exports
    │
    ├── build_corpus.py                   # (laptop) Wikipedia → chunked embeddings → corpus.db
    ├── build_blurbs.py                   # (laptop) Ollama → structured blurbs per species
    ├── compile_blurbs.py                 # (laptop) write blurb_json into corpus.db
    ├── generate_prose.py                 # (laptop) prose refinement pass over blurbs
    ├── export_tflite.py                  # (laptop) Keras → TFLite (int8/dynamic/fp16/fp32) + manifest
    ├── export_int8_tflite_subprocess.py  # (laptop) PyTorch/ONNX → INT8 TFLite via onnx2tf
    │
    └── tests/
        ├── test_corpus.py                # retrieval (species-partitioned KNN) + open_db()
        ├── retrieval_recall_check.py     # recall of legacy vs partitioned retrieval
        ├── eval_tflite.py                # device-path model eval; fits temperature + threshold
        ├── test_pipeline_integration.py  # pipeline plumbing, every path
        ├── test_intent_units.py          # intent classifier unit tests
        ├── test_phrasing.py              # prompt phrasing regression tests
        ├── eval_pipeline.py              # pipeline eval (path distribution, accuracy)
        ├── eval_harness.py               # retrieval threshold tuning
        ├── eval_e2e.py                   # end-to-end answer quality eval
        ├── eval_intent.py                # intent classifier eval
        ├── diff_runs.py                  # compare two eval result sets
        ├── verify_matches_db.py          # corpus coverage per species
        ├── preview_danger_prompts.py     # inspect built danger prompts
        └── mem_check.py                  # RSS memory tracking (called in voice loop)
```

---

## Key Tuning Constants

| Constant | Location | Default | Effect |
|---|---|---|---|
| `id_threshold` | `outputs/deploy/model_manifest.json` | fit by `eval_tflite.py` | Minimum TTA-averaged confidence to accept an identification at startup (`--id-threshold` overrides) |
| `temperature` | `outputs/deploy/model_manifest.json` | fit by `eval_tflite.py` | Logit temperature, fit on the exported model's TTA-averaged val output |
| `CAMERA_WARMUP_FRAMES` | `full_roboranger_run.py` | 5 | Frames flushed before capture to clear stale V4L2 buffer |
| `CONF_THRESH` | `live_inference_mv3.py` | 0.15 | Display threshold for the standalone camera loop (below = grey text) |
| `INFER_HZ` | `live_inference_mv3.py` | 2.0 | Camera inference rate in the standalone loop. TTA runs 4 passes per tick. |
| `DEFAULT_RETRIEVAL_THRESHOLD` | `pipeline.py` | 0.55 | Cosine similarity floor for RAG chunks |
| `RETRIEVAL_K` | `pipeline.py` | 5 | Chunks retrieved from the species' partition (prompt keeps `MAX_CHUNKS = 3`) |
| `DIRECT_ROUTE_SCORE_THRESHOLD` | `pipeline.py` | 0.65 | Minimum intent score for medium-confidence direct routing (skips LLM) |
| `WildlifeGate.ACCEPT_MARGIN` | `wildlife_gate.py` | 0.05 | Wildlife–off-topic margin above which gate accepts with high confidence |
| `SAMPLING["max_tokens"]` | `llm_backends.py` | 100 | LLM output cap for both backends (tuned for UNO Q speed; 200 was too slow) |
| `VEC_CHUNK_SIZE` | `corpus_schema.py` | 16 | vec0 slots per partition chunk (DB size vs. scan granularity) |
| `EMBED_MODEL` | `corpus_schema.py` | `BAAI/bge-small-en-v1.5` | Must match the model used at corpus build time (checked at startup) |
