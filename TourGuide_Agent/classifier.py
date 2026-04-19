"""Inference wrappers for the La Jolla Cove species classifier.

Two implementations with the same contract (`Prediction` output shape + a
`predict(image_path, top_k)` method + a `device` attribute):

- `SpeciesClassifier` — loads the local DINOv2 checkpoint from
  `species_identification/outputs/best_model.pth` and runs inference on CPU/MPS/CUDA.
  Architecture and preprocessing match `species_identification.ipynb`.
- `RemoteSpeciesClassifier` — POSTs the image to a remote HTTP endpoint
  (typically the Modal deployment in `serving/modal_app.py`) and maps the
  response back to `Prediction` objects. Common names are still resolved
  locally from the iNaturalist CSV since the remote endpoint only returns
  scientific names.

Pick which one to use with `load_classifier()` — set `CLASSIFIER_URL` in the
environment (or `TourGuide_Agent/.env`) to route to the remote API; leave it
unset to use the local checkpoint.

Usage:

    from classifier import load_classifier

    clf = load_classifier()                 # picks local vs remote from env
    for p in clf.predict("photo.jpg", top_k=3):
        print(f"{p.probability:.1%}  {p.scientific_name}  ({p.common_name})")
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = _REPO_ROOT / "species_identification" / "outputs" / "best_model.pth"
DEFAULT_CSV = _REPO_ROOT / "observations-711999" / "observations-711999.csv"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class _DINOv2Classifier(nn.Module):
    """Architecture mirror of species_identification.ipynb's classifier."""

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


@dataclass
class Prediction:
    scientific_name: str  # space form, e.g. "Zalophus californianus"
    common_name: str | None
    probability: float


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_common_names(csv_path: Path) -> dict[str, str]:
    """Build a scientific_name (underscore form) → common_name mapping.

    Keyed by underscore form to match checkpoint class_names. Returns an
    empty dict if the CSV is missing.
    """
    mapping: dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sci = (row.get("scientific_name") or "").strip()
            common = (row.get("common_name") or "").strip()
            if not sci or not common:
                continue
            mapping.setdefault(sci.replace(" ", "_"), common)
    return mapping


class SpeciesClassifier:
    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        image_size: int,
        device: torch.device,
        common_names: dict[str, str],
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
        self.device = device
        self.common_names = common_names
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        csv_path: Path | str = DEFAULT_CSV,
        device: torch.device | None = None,
    ) -> "SpeciesClassifier":
        checkpoint_path = Path(checkpoint_path)
        csv_path = Path(csv_path)
        device = device or _pick_device()

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = checkpoint["model_name"]
        class_names = checkpoint["class_names"]
        image_size = checkpoint["image_size"]
        embed_dim = _EMBED_DIMS[model_name]

        backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        model = _DINOv2Classifier(backbone, embed_dim, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        return cls(
            model=model,
            class_names=class_names,
            image_size=image_size,
            device=device,
            common_names=_load_common_names(csv_path),
        )

    def predict(self, image_path: Path | str, top_k: int = 3) -> list[Prediction]:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        probs = torch.softmax(logits.float(), dim=1).squeeze(0)
        k = min(top_k, probs.numel())
        top_probs, top_idx = probs.topk(k)
        results: list[Prediction] = []
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
            label = self.class_names[idx]  # underscore form
            results.append(Prediction(
                scientific_name=label.replace("_", " "),
                common_name=self.common_names.get(label),
                probability=float(prob),
            ))
        return results


class RemoteSpeciesClassifier:
    """HTTP client for a remote classifier (e.g. the Modal deployment).

    Matches the `SpeciesClassifier` contract so it can be dropped in without
    changing anything in `demo.py` / `TourSession`. The remote endpoint is
    expected to accept a POST with raw image bytes as the body and `top_k` as
    a query param, returning `{"predictions": [{"scientific_name": str,
    "probability": float}, ...]}`.

    Common names are resolved locally from the iNaturalist CSV since the
    endpoint only returns scientific names.
    """

    def __init__(
        self,
        url: str,
        common_names: dict[str, str],
        *,
        timeout: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.common_names = common_names
        self.timeout = timeout
        # Displayed by demo.py; kept short so "Classifier ready on ..." stays readable.
        self.device = f"remote ({self.url})"

    @classmethod
    def from_url(
        cls,
        url: str,
        csv_path: Path | str = DEFAULT_CSV,
        *,
        timeout: float = 30.0,
    ) -> "RemoteSpeciesClassifier":
        return cls(
            url=url,
            common_names=_load_common_names(Path(csv_path)),
            timeout=timeout,
        )

    def predict(self, image_path: Path | str, top_k: int = 3) -> list[Prediction]:
        # Lazy import so users without `requests` can still use the local classifier.
        import requests

        image_bytes = Path(image_path).read_bytes()
        resp = requests.post(
            self.url,
            params={"top_k": top_k},
            data=image_bytes,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[Prediction] = []
        for entry in data.get("predictions", []):
            sci = entry["scientific_name"]            # space form
            key = sci.replace(" ", "_")               # underscore form for CSV lookup
            results.append(Prediction(
                scientific_name=sci,
                common_name=self.common_names.get(key),
                probability=float(entry["probability"]),
            ))
        return results


def load_classifier(
    *,
    url: str | None = None,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    csv_path: Path | str = DEFAULT_CSV,
) -> SpeciesClassifier | RemoteSpeciesClassifier:
    """Factory: pick remote vs local based on the `CLASSIFIER_URL` env var.

    Set `CLASSIFIER_URL=https://...modal.run` in `TourGuide_Agent/.env` to
    route through the Modal deployment. Leave it unset (or empty) to load the
    local PyTorch checkpoint. The explicit `url` arg overrides the env var.
    """
    remote_url = url or os.environ.get("CLASSIFIER_URL") or None
    if remote_url:
        return RemoteSpeciesClassifier.from_url(remote_url, csv_path=csv_path)
    return SpeciesClassifier.load(checkpoint_path=checkpoint_path, csv_path=csv_path)
