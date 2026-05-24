# RoboRanger — Project Summary

RoboRanger is a portable, hardware-based eco tour guide built for DataHacks 2026. Point it at anything in nature and it tells you what it's looking at — in the voice of a national park ranger — and answers follow-up questions about the animal or plant. The demo target is **La Jolla Cove**, but the system is designed to work anywhere.

Target hardware: **Arduino UNO Q** (Linux Debian, quad-core SoC, Python 3.11).

Devpost: https://devpost.com/software/roboranger

---

## Two Architectures, One Product

RoboRanger was built in two distinct phases that reflect how the project's constraints evolved.

### Architecture 1 — Network Agent

The first architecture offloaded all intelligence to the cloud. A lightweight Arduino client captured camera frames and microphone audio, sent them over HTTP to a FastAPI server, and played back the TTS audio it received. The server ran the species classifier (DINOv2 ViT-B/14) and the ranger agent (Claude Haiku via the Anthropic API), with ElevenLabs handling both transcription and text-to-speech.

This got a working demo quickly and produced high-quality narration, but it required an internet connection, an ngrok tunnel, and API keys at runtime for ElevenLabs TTS and STT.

→ See [networkAgentREADME.md](networkAgentREADME.md) for full details on this architecture.

### Architecture 2 — On-Device Agent

The second architecture moved everything onto the device. No cloud, no API keys, no connectivity required after setup. The tradeoffs were significant — a 360M-parameter local LLM instead of Claude Haiku, whisper.cpp instead of ElevenLabs Scribe, Piper instead of ElevenLabs TTS — and the engineering challenge was making those tradeoffs feel invisible to a visitor asking natural questions.

→ See [onDeviceAgentREADME.md](onDeviceAgentREADME.md) for full details on this architecture, including build workflow, device setup, and how to run it.

---

## What the On-Device Agent Does

The final system runs entirely on the UNO Q. A visitor points the device at an animal or plant. It identifies what it sees, announces it, then takes spoken questions and answers them — all locally, in a few seconds per turn.

```
Camera (one frame at startup)
    └─→ MobileNetV3 INT8 TFLite  ─→  species_id
                                           │
Microphone (push-to-talk per question)     │
    └─→ Whisper.cpp (ggml-tiny.en)         │
              │                            │
              ▼                            ▼
         query text    ──→    RoboRangerPipeline
                                    │
              ┌─────────────────────┤
              │                     │
         Direct route         RAG + LLM
         (blurb fields,       (sqlite-vec retrieval +
          sub-ms, no LLM)      SmolLM2-360M via llama.cpp)
              │                     │
              └─────────────────────┘
                          │
                    Piper TTS → aplay
```

The entry point is `full_roboranger_run.py`. Species identification happens **once at startup** — the camera is opened, one frame is classified, the result is announced, and the camera is released. All subsequent turns are voice-only. This keeps per-turn latency predictable and avoids holding the camera handle in the voice loop.

---

## Key Engineering Decisions

### Why move off the cloud at all?

The network agent worked, but connectivity is unreliable at a park. The demo also felt more compelling — and the science more honest — when the device genuinely operated on its own. A ranger in the field doesn't need a cell signal to know what a sea lion eats.

### Vision: MobileNetV3 over DINOv2

DINOv2 ViT-B/14 was the right backbone for training accuracy but too large to run at any useful rate on the UNO Q. MobileNetV3 (particularly the Small variant) runs well above real-time with TTA — 4 views per frame at 2 Hz. The accuracy gap was closed with two training techniques:

- **Two-phase training**: freeze the backbone and train only the classification head first, then unfreeze the top blocks and fine-tune at a lower rate. With ~20 images per class, training the head before unfreezing prevents the backbone from being destroyed before the head stabilizes.
- **Temperature scaling**: after training, a scalar `T` is fitted on the validation set so that the model's softmax outputs are calibrated probabilities rather than raw overconfident scores. This matters for the 0.55 confidence gate at session start.

**Test-Time Augmentation** (4 views per frame: original, horizontal flip, 90% crop, 80% crop) meaningfully improves top-1 accuracy at no extra model cost — just 4× the interpreter invocations per classification.

### Knowledge: offline RAG instead of bigger weights

Fitting La Jolla Cove species knowledge into a 360M-parameter model's weights would require fine-tuning and would still produce hallucinations on edge cases. Instead, a `corpus.db` SQLite database is built on the laptop and shipped to the device. It holds:

- **Structured blurbs** — typed JSON records per species (appearance, diet, habitat, behavior, danger to humans/pets, notable facts), generated from Wikipedia source text by a local Ollama model. Danger fields are enums (`"no"` / `"mild"` / `"yes"`), not free text, because small models are far more reliable on labeled values than on interpreting hedged prose.
- **Wikipedia text chunks** — full paragraphs embedded with `bge-small-en-v1.5` and stored as vectors in a `sqlite-vec` virtual table for approximate nearest-neighbor search.

For the majority of visitor questions ("what does it eat?", "is it dangerous?"), the answer comes entirely from the structured blurb — no retrieval, no LLM. Only complex or open-ended questions go to SmolLM2.

### Query routing: four stages before the LLM

The pipeline is designed so the LLM is the last resort, not the first. Four stages run before any generation:

1. **WildlifeGate** — rejects off-topic questions (bathrooms, parking, chitchat) using a relative comparison between two embedding centroids. Any positive margin toward the wildlife centroid → accept. A relative comparison is essential because off-topic queries like "where is the X" score high against wildlife intents in absolute terms; they only look off-topic when directly compared to an off-topic centroid.

2. **BlurbStore** — SQLite lookup. If the species isn't in the corpus, the pipeline refuses cleanly. This is better than letting the LLM hallucinate from zero grounding.

3. **IntentClassifier** — 8-way classification (DESCRIPTION, DANGER, DIET, HABITAT, SIZE, BEHAVIOR, IDENTIFICATION, CONSERVATION) using the same `bge-small` embedder with per-intent prototype centroids. No second model load.

4. **Direct route** — for high-confidence simple intents, the answer is formatted from blurb fields in code and returned without retrieval or LLM. Sub-millisecond. This is where "is it dangerous?" and "what does it eat?" are handled for the vast majority of turns.

### Prompt engineering for a 360M model

SmolLM2-360M follows short, declarative instructions well but struggles with implicit reasoning tasks. Two specific interventions were needed:

**Danger polarity**: visitors phrase risk questions in opposite ways ("is it dangerous?" vs. "is it safe to touch?"). The model cannot reliably flip "yes"/"no" based on question framing. The prompt builder detects the question's polarity and rephrases both the verdict prose and the directive lead word to match — so the model only needs to paraphrase the provided text, not reason about the flip.

**Verdict blocks**: for DANGER intent, the danger fields are rendered as a plain-English `VERDICT:` sentence at the top of the species facts block, set off by blank lines. Small models track section breaks better than inline modifiers, and leading with the verdict means the answer is correct even if the model truncates early.

### Latency: hiding load time

The UNO Q is slow to load models. Three strategies hide this from the visitor:

- **Piper loads first**, before anything else, so the device can speak error messages during the rest of startup rather than going silent.
- **Whisper loads in a background thread** while the camera is capturing and classifying. By the time identification finishes, Whisper is ready.
- **llama.cpp `--cache-prompt`** caches the system prompt and species blurb prefix across turns. Only the user's question is re-encoded per turn, not the full context.

---

## What Was Used (On-Device)

| Component | Technology |
|---|---|
| Vision classifier | MobileNetV3-Small (TF/Keras), INT8 TFLite via onnx2tf |
| Species knowledge | SQLite + sqlite-vec, bge-small-en-v1.5 embeddings |
| LLM | SmolLM2-360M-Instruct (q8_0 GGUF), served via llama.cpp |
| STT | whisper.cpp (ggml-tiny.en-q5_1.bin) via pywhispercpp |
| TTS | Piper TTS (en_US-lessac-low.onnx) |
| Blurb generation | Ollama (laptop-side build step, not on device) |

## What Was Used (Network Agent)

| Component | Technology |
|---|---|
| Vision classifier | DINOv2 ViT-B/14, fine-tuned on iNaturalist La Jolla data |
| Ranger agent | Claude Haiku (`claude-haiku-4-5-20251001`) via Anthropic API |
| STT | ElevenLabs Scribe (`scribe_v1`) |
| TTS | ElevenLabs (`eleven_turbo_v2_5`, voice "Brian") |
| Server | FastAPI + uvicorn, tunneled via ngrok |
