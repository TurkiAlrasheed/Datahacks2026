# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project (DataHacks 2026)

A wildlife / eco tour guide for the **Arduino UNO Q** (Debian + Python 3 on the SoC), so any new runtime dependency must be pip-installable there. Two architectures share the repo:

- **RoboRanger — on-device, the current focus.** Repo-root scripts + `species_identification/`. Fully offline: TFLite camera classifier → push-to-talk Whisper → embedding gate/intent + sqlite-vec RAG → SmolLM2 on llama.cpp → Piper TTS. The classifier covers 50 San Diego (UCSD iNaturalist) species + `seashore` = 51 classes. `onDeviceAgentREADME.md` has the long form (measurements, OTA design, device setup); where it disagrees with the code, trust the code.
- **TourGuide_Agent — cloud/network.** Claude ranger persona + ElevenLabs voice, backed by a DINOv2 classifier trained in notebooks on La Jolla Cove iNaturalist data.

## Environment

- **Python:** `C:\Users\azizk\miniconda3\envs\Datahacks2026V2\python.exe` (conda env `Datahacks2026V2`, Python 3.11; newer versions break the deps). System python has none of the deps.
- Deps: `requirements.txt` = device runtime, pinned (`ai-edge-litert==2.1.4`, `sqlite-vec==0.1.9`). `requirements-build.txt` = laptop build (TensorFlow <2.22, sqlite-vec ≥0.1.6 for `PARTITION KEY`).
- Run laptop tools **from the repo root**: `export_tflite.py`, `eval_tflite.py` and `retrieval_recall_check.py` default to repo-relative paths (`species_identification/cnn/_splits`, `species_identification/outputs/.eval_cache`, `species_identification/offline-info/corpus.db`). Entry points, pipeline modules and tests resolve imports from `__file__`, so they work from any cwd.
- LiteRT evaluation is slow on this laptop: pass `eval_tflite.py --workers N` (one interpreter per process). Logits are cached per model-file hash, so reruns are instant.
- Windows dev box; the Bash tool is Git Bash (forward slashes, `/dev/null`).

## RoboRanger (on-device)

### Entry points (repo root)

| Script | Purpose |
|---|---|
| `full_roboranger_run.py` | Main loop: camera ID once, then push-to-talk Q&A. Needs `--voice <piper .onnx>` and `--whisper-model <ggml .bin>`. `--backend llama-cpp` on the device (default `ollama`). `--model-dir` (default `species_identification/outputs/deploy`), `--species X` skips the camera, `--id-threshold` overrides the manifest, `--save-id-frames DIR`. `--camera-test` grabs one frame and exits without loading (or needing) any model. |
| `run_pipeline.py` | Text-only pipeline driver, no audio. `--species` plus exactly one of `--query "..."`, `--repl`, `--profile <queries.txt>` (p50/p95 per stage). |
| `voice_loop.py` | Push-to-talk loop with no camera; `--species`, `--voice`, `--whisper-model` required. |
| `live_inference_mv3.py` | Headless camera loop for any `--model-dir` (default `outputs/deploy`): 2 Hz, TTA, logs to `species_identification/outputs/live/<model dir name>/`. `--camera-index` defaults to **1** (laptop external webcam; the UNO Q is 0). |
| `live_inference_mv2.py` | Same `main()` with `--model-dir` defaulting to `outputs/mobilenetv2_qat`. |

`full_roboranger_run.py` startup: load Piper first (so failures can be spoken) → corpus preflight (`check_corpus_schema`) → start the Whisper load on a background thread → load the classifier and warn about classes with no blurb → on Enter, flush 5 frames, grab one and call `classifier.identify()` → speak the result; if refused (negative class, below threshold, camera error), say why and wait for Enter again → build the pipeline on the main thread (sqlite connections are thread-bound) → join Whisper → `pipeline.warmup(species)` (real retrieval + one LLM call) → write `device_status.json` (corpus version/hash, tflite sha256, T, threshold) → voice loop. `record_utterance()` is the one function to swap for a GPIO button.

### Q&A pipeline

`RoboRangerPipeline.answer(species_id, query) -> Response` in `species_identification/pipeline/pipeline.py` is pure (no I/O; the entry points own audio and camera). Construct it with `pipeline_factory.build_pipeline()`, which checks the corpus schema before loading the embedder. Stages: `WildlifeGate` → `BlurbStore.get` → `IntentClassifier` → direct route (confident simple intents answered from blurb fields, no LLM) → retrieval (`RETRIEVAL_K = 5`, drop cosine < `DEFAULT_RETRIEVAL_THRESHOLD = 0.55`) → `prompt_builder.build_messages` → `LlamaCppBackend` (localhost:8080) or `OllamaBackend` (localhost:11434), `max_tokens` 100. `Response.path` is one of `gate_rejected`, `species_not_found`, `intent_unclear`, `blurb_direct`, `blurb_only`, `blurb_plus_chunks`, `llm_error`. Gate, intent, prompt builder and backends live in `species_identification/llm-tuning/` (hyphenated, so everything imports via `sys.path` inserts, not packages).

### Vision: one manifest-driven TFLite classifier

- `species_identification/vision/tflite_classifier.py` (`TFLiteClassifierTTA`) is the only classifier the device uses. Don't reintroduce per-script preprocessing.
- Every model dir holds `model_*.tflite` + `model_manifest.json`: `model_file`, `arch`, `input_range` (`raw_0_255` | `minus1_1` | `imagenet`), `img_size`, `input_dtype` (`uint8` | `int8` | `float32`), `class_names` (logit order), `negative_classes` (`["seashore"]`), `temperature`, `id_threshold`, `tta_views`, `source`, `eval`. Unknown keys, or a tensor shape / dtype / class count that disagrees with the manifest, raise `ModelContractError`. A wrong `input_range` does **not** crash; it silently yields confident garbage, which is why the domain travels with the model.
- `predict()` averages temperature-scaled softmax over 4 TTA views; `identify()` applies the policy (negative class, or conf < `id_threshold` → refuse). Interpreters default to reference kernels (XNNPACK off) to match the UNO Q.
- **The device runs `species_identification/outputs/deploy/`**: MobileNetV3-Large, dynamic-range quantized (`model_dynamic.tflite`, float32 I/O, `raw_0_255`, 320 px, T = 1.0, `id_threshold` 0.63; test TTA top-1 79.5%, precision 91.1% at threshold), copied from `outputs/mobilenetv3l_dynamic/`. Calibrated fallback: `outputs/mobilenetv2_qat/` (full int8, `minus1_1`).
- **Full-int8 PTQ collapses MobileNetV3-Large** (hard-swish/SE activations; ~17% top-1 per the deploy manifest's notes). Export MV3 with `--quant dynamic`. MobileNetV2 is quantization-aware trained, so int8 is fine for it.

### Train → export → evaluate → deploy

1. **Train:** `species_identification/cnn/mobile_net_v3l.py` | `mobile_net_v3s.py` | `mobile_net_v2.py`. Data: `species_identification/ucsd-data/`; splits are rebuilt into `cnn/_splits/` on every run. Each writes to `outputs/<arch>/` (`mobilenetv3l`, `mobilenetv3s`, `mobilenetv2_qat`), never the shared `outputs/` root. Each ends with an **int8** export into that dir, so re-export MV3 as dynamic.
   - **Float-model diagnostics:** `python species_identification/cnn/model_diagnostics.py --arch <arch>` (`--arch` is required). It defaults to `outputs/<arch>/final.keras`, because that's the model `temperature.json` is fit on and the one the trainers export. It reads `class_names.json` and `temperature.json` from the same dir and writes `outputs/<arch>/diagnostics/`. Input scaling comes from `export_tflite.ARCH_INPUT_RANGE` via the device's `normalize()`, and image size from the model. It exits if the model's in-graph `Rescaling` contradicts `--arch`. `mobilenetv2_qat` loads under `TF_USE_LEGACY_KERAS=1` inside tfmot's `quantize_scope()`.
2. **Export (Keras):** `python species_identification/export_tflite.py --keras <.keras> --arch <arch> --quant int8|dynamic|float16|float32 --out <dir>` (default `int8`). Writes `model_<quant>.tflite` + a manifest with placeholder T/threshold and round-trips it through `TFLiteClassifierTTA`. PyTorch/DINOv2 models use `export_int8_tflite_subprocess.py` (ONNX → onnx2tf, `imagenet` manifest).
3. **Evaluate + calibrate:** `python species_identification/tests/eval_tflite.py --model-dir <dir> [--model-dir <dir2>] --workers 5 --write`. Runs the real device path on `cnn/_splits` val/test. `--write` stores T (min val NLL) and `id_threshold` (lowest with val precision ≥ `--target-precision`, default 0.90) plus an `eval` summary in the manifest. Also: `--keras` (float-model agreement, single model dir), `--limit-per-class`, `--ref-parity N`, `--input-range` (forces a wrong domain; never written).
4. **Deploy:** copy the chosen model dir to `outputs/deploy/`.

Loose files at the `outputs/` root (`best.keras`, `model_int8.tflite`, `temperature.json`, …) date from when all trainers shared one folder and overwrote each other. Don't take calibration from them. `outputs/best_model.pth` is the DINOv2 checkpoint TourGuide_Agent loads.

### Corpus (`species_identification/offline-info/corpus.db`)

- `species_identification/corpus_schema.py` is the one schema definition shared by builders, runtime and tests (stdlib + a sqlite-vec connection only, so the device can import it). **Schema v2:** `species` (incl. `blurb_json`), `chunks`, `chunk_vectors USING vec0(species_id TEXT PARTITION KEY, category TEXT, embedding float[384], chunk_size = 16)`, plus `corpus_meta` (schema/corpus version, `content_sha256`, embed model, sqlite-vec version, counts) and `species_manifest` (per-species sha256). Call `refresh_manifest()` after any write. `EMBED_MODEL = "BAAI/bge-small-en-v1.5"`.
- The runtime refuses a v1 corpus or a different embedder: `check_corpus_schema()` (called by `build_pipeline` and the `full_roboranger_run.py` preflight) raises `CorpusSchemaError`. Migrate with `python species_identification/migrate_corpus_v2.py <old.db> <new.db>` (copies vectors byte-for-byte and verifies; `--measure` compares chunk sizes).
- Build order: `build_corpus.py <species.json> <corpus.db>` → `build_blurbs.py <species.json> <blurbs.yaml>` (local Ollama; review by hand) → `compile_blurbs.py <blurbs.yaml> <corpus.db>` (writes `blurb_json` + a `category='blurb'` chunk).
- **Retrieval is production code in `species_identification/tests/test_corpus.py`** (imported by `pipeline.py`). `knn_for_species(conn, q_bytes, species_id, k)` is the partition-scoped exact KNN with `category != 'blurb'` (the blurb is already in the prompt as SPECIES FACTS). `retrieve(conn, embedder, species_id, query, k)` adds pronoun → common-name query rewriting and the bge query prefix, returning `(category, text, l2_distance)`.
- Measure recall/latency: `python species_identification/tests/retrieval_recall_check.py --db species_identification/offline-info/corpus.db [--mode legacy|partitioned|both] [--no-rewrite] [--sweep]`.

### Tests (run from any cwd)

```bash
python species_identification/tests/test_pipeline_integration.py    # stub embedder/LLM; every Response path, v1 refusal, warmup; exit 1 on failure
python -m pytest species_identification/tests/test_intent_units.py   # intent classifier, wildlife gate, prompt builder
```

The `tests/eval_*.py` scripts are quality evals that need real models or backends, not unit tests.

## TourGuide_Agent (cloud)

- `classifier.py`: `SpeciesClassifier.load()` rebuilds DINOv2 from `species_identification/outputs/best_model.pth` metadata (`model_name`, `class_names`, `image_size`); common names come from `observations-711999/observations-711999.csv`. `load_classifier()` returns `RemoteSpeciesClassifier` instead when `CLASSIFIER_URL` is set (Modal deployment).
- `location.py`: `get_location()` tries explicit lat/lon → GPS on `GPS_PORT` → `ipapi.co` → unknown, then reverse-geocodes via Nominatim. Short timeouts, silent fallback.
- `tour_guide.py`: `TourSession` (`see` / `ask` / `look_at`) remembers sightings and narrates NEW vs REPEAT. Location-general prompt; calls `get_location()` at init when no location is given (one-time network delay). Model `claude-haiku-4-5-20251001`, system prompt ephemerally cached, season hard-coded to Spring.
- `voice.py`: ElevenLabs STT `scribe_v1` / TTS `eleven_turbo_v2_5`, interruptible playback; needs `ELEVEN_LABS_API_KEY`.
- `demo.py`: REPL (`see <image>`, Enter = voice turn, text = question, `--no-voice`). Frames below `CONFIDENCE_THRESHOLD = 0.7` or in `NEGATIVE_CLASSES = {"seashore"}` go to `look_at()` instead of `see()`.
- `server.py`: FastAPI wrapper (`/session`, `/see`, `/ask`, `/look`, `/tts`, `/stt`, `/health`); start with `uvicorn server:app` from `TourGuide_Agent/`. `arduino_client.py` is the thin UNO Q client for it (`SERVER_URL`). `keypad.py` reads a 4×4 keypad via sysfs GPIO. `generate_audio.py` does one-off ElevenLabs TTS to a file.
- Run: `pip install -r TourGuide_Agent/requirements.txt`, put `ANTHROPIC_API_KEY` and `ELEVEN_LABS_API_KEY` in `TourGuide_Agent/.env`, then `python TourGuide_Agent/demo.py [image]`.

## DINOv2 notebooks (La Jolla classifier for TourGuide_Agent)

Run in order; they hand off via the filesystem. `species_identification/data_preparation.ipynb` takes the top species from the iNaturalist CSV, downloads to `images_raw/` (a cache), and makes a stratified 70/15/15 split into `data/{train,val,test}/<Scientific_name>/`. `species_identification/species_identification.ipynb` trains a linear head on a frozen DINOv2 ViT-B/14 (`FREEZE_BACKBONE = False` switches to fine-tuning at `LR_FINETUNE`). It writes `outputs/best_model.pth` (state dict + `class_names`, `image_size`, `model_name`), `class_names.json`, `metrics.json` and `training_curves.png`. This classifier is a deliberate La Jolla Cove specialist (biased toward sea lions, cormorants, gulls, pelicans); don't broaden its data. `CSV_PATH` uses backslashes and triggers a harmless `SyntaxWarning`.

## Conventions

- Class label = `scientific_name.replace(" ", "_")`. Always restore class order from the checkpoint or manifest, never from a directory listing.
- `seashore` is a negative "nothing to narrate" class in both architectures: a regular class for training, never announced at inference.
- Gitignored, derived data: `images_raw/`, `data/`, `ucsd-data/`, `species_identification/outputs/.eval_cache/`, `species_identification/outputs/live/`, `device_status.json`. `.gitignore` still names the stale `observations-711989`; the real dir is `observations-711999`.
