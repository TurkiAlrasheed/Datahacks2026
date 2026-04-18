# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Eco Tour Guide (DataHacks 2026)

A hardware-based eco tour guide. The device observes the user's surroundings (camera), combines visual input with location + seasonality, identifies flora and fauna, and narrates a national-park-ranger-style tour. **The hackathon demo target is La Jolla Cove** — both the species model and the agent's knowledge are scoped to that locale.

The repo has two workstreams that feed the same product:

- **`species_identification/`** — trains the on-device / cloud image classifier. Scope is intentionally narrow: the species commonly observed at La Jolla Cove, sourced from iNaturalist. The resulting model is the "eyes" of the tour guide.
- **`TourGuide_Agent/`** — the AI agent layer (currently empty scaffold). Takes species predictions + GPS + date/season as context and generates the ranger-style tour narration. This is where the LLM prompting, tool use, and any TTS/device integration will live.

Input data: `observations-711999/observations-711999.csv` — an iNaturalist export for the La Jolla Cove area (id, image_url, scientific_name, common_name, lat/lon, observed_on, etc.). Lat/lon and `observed_on` are the hooks for the agent's location/seasonality reasoning.

## species_identification pipeline: raw CSV → trained model

Two notebooks run in order and hand off via filesystem, not variables:

1. **`data_preparation.ipynb`** reads `../observations-711999/observations-711999.csv`, keeps the top-N most common species at La Jolla Cove (default: top 50, min 15 images/class → ~40 classes), rewrites iNaturalist image URLs to a chosen size variant (`medium` → `large`), parallel-downloads into `images_raw/` (cached — safe to re-run), does a stratified per-species 70/15/15 split, then copies images into `data/{train,val,test}/<Scientific_name>/` — the `ImageFolder` layout the trainer expects.

2. **`species_identification.ipynb`** loads `data/` with `torchvision.datasets.ImageFolder`, attaches a linear head to a frozen DINOv2 ViT-B/14 backbone (`torch.hub.load("facebookresearch/dinov2", ...)`), trains with AdamW + cosine LR + AMP + early stopping, and writes artifacts to `outputs/`:
   - `best_model.pth` — state dict + `class_names`, `image_size`, `model_name` metadata (this is what the agent will load)
   - `class_names.json`, `metrics.json` (full history), `training_curves.png`

Toggling `FREEZE_BACKBONE = False` switches from linear probing to full fine-tuning and automatically uses the lower `LR_FINETUNE` (1e-5 vs 1e-3).

Because the demo is La Jolla Cove only, it's fine — expected, even — that the class list is a small, biased subset (heavily weighted toward sea lions, cormorants, gulls, pelicans). Do not "fix" this imbalance by sampling broader geography; the model is meant to be a local expert, not a general species classifier.

## TourGuide_Agent (to be built)

Conceptual contract the agent is expected to honor:

- **Inputs**: image (or precomputed top-k species predictions from the classifier), GPS coordinates, timestamp (→ season), optionally user's stated interests.
- **Behavior**: park-ranger persona — warm, conversational, informative. Blends species facts with site-specific context (tides, breeding seasons, recent sightings, La Jolla-specific history/geology).
- **Outputs**: narration text (and eventually speech for the hardware device).

When implementing here, prefer loading the classifier checkpoint via the metadata saved in `best_model.pth` (`class_names`, `image_size`, `model_name`) rather than re-deriving from a directory listing — training and inference must agree on class order.

## Key conventions

- **Class label = `scientific_name.replace(" ", "_")`**. `ImageFolder` sorts alphabetically, so `class_names` order is deterministic and must match between training and inference — always restore class names from the checkpoint.
- **Image size variants** are swapped via URL token substitution in `upgrade_url()` (`/medium.jpg` → `/large.jpg`). iNaturalist supports: `square`, `small`, `medium`, `large`, `original`.
- **`images_raw/` is a download cache** keyed by observation `id`. It is gitignored and can be deleted; `data_preparation.ipynb` will re-download. `data/` is likewise derived and gitignored — never commit either.
- **Device selection** auto-detects CUDA → MPS → CPU. AMP (`torch.amp.autocast`/`GradScaler`) is CUDA-only and controlled by `USE_AMP`.
- **Backbone stays in `eval()` during training** when frozen — the train loop explicitly re-asserts this after `model.train()` to keep norm/dropout layers quiet.
- **Scope discipline**: if you're tempted to broaden the dataset, generalize the model, or make the agent work "anywhere" — don't. The hackathon bet is a polished La Jolla Cove demo, not a general-purpose system.

## Environment notes

- Developed on Windows with CUDA 11.8, PyTorch 2.6, torchvision 0.21, RTX 4070 Laptop GPU. Shell is bash (Git Bash / WSL-style), so use forward slashes and `/dev/null`.
- `CSV_PATH` in `data_preparation.ipynb` uses a Windows-style relative path (`..\observations-711999\observations-711999.csv`) which Python parses with a `SyntaxWarning: invalid escape sequence`. Harmless but worth fixing to a raw string or forward slashes if touched.
- `.gitignore` references `observations-711989` but the actual directory is `observations-711999` — the ignore rule is stale. The raw/processed image dirs are still correctly ignored via their own entries.

## Running

No package manifest, build system, or tests exist yet. Notebooks are the current entrypoint — run cells top-to-bottom in Jupyter / VS Code. Dependencies inferred from imports: `pandas`, `requests`, `Pillow`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`. TourGuide_Agent dependencies will be added as that workstream comes online.
