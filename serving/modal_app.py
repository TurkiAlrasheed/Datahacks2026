"""Modal deployment for the La Jolla species classifier.

Serves the DINOv2 ViT-B/14 + linear-head checkpoint
(`species_identification/outputs/best_model.pth`) behind a single public HTTP
endpoint. Model weights live in a Modal Volume (not in the image) so
re-deploying code doesn't re-upload the checkpoint.

One-time setup (run from the repo root):

    pip install modal
    modal setup                                                        # authenticate
    modal volume create species-ckpt                                   # create the volume
    modal volume put species-ckpt \\
        species_identification/outputs/best_model.pth \\
        /best_model.pth                                                # upload checkpoint

Deploy (and redeploy on code changes):

    modal deploy serving/modal_app.py

Modal prints the public URL on deploy, e.g.:

    https://<your-workspace>--la-jolla-classifier-classifier-predict.modal.run

Invoke (client POSTs raw image bytes, not multipart):

    curl -X POST --data-binary @photo.jpg \\
        "https://<your-url>?top_k=3"

Returns JSON:

    {
        "model": "dinov2_vitb14",
        "predictions": [
            {"scientific_name": "Zalophus californianus", "probability": 0.97},
            {"scientific_name": "Phoca vitulina",         "probability": 0.02},
            ...
        ]
    }

Notes
- Cold start ~8-15 s on CPU (model load + checkpoint from volume; the DINOv2
  backbone is baked into the image so there's no network download at start).
- `scaledown_window=300` keeps the container warm for 5 min after the last
  request, so back-to-back calls are fast.
- Switch to `gpu="T4"` below for ~100 ms-per-request inference (costs while warm).
- The endpoint is public. For production, wrap with Modal's `secret=` auth or
  put a proxy in front.
"""

from __future__ import annotations

import io

import modal
from fastapi import HTTPException, Request


APP_NAME = "la-jolla-classifier"
VOLUME_NAME = "species-ckpt"


app = modal.App(APP_NAME)


def _prefetch_dinov2() -> None:
    """Download the DINOv2 backbone at image-build time so cold starts skip it."""
    import torch
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install numpy in its own layer FIRST so subsequent layers see it and
    # torch's wheel finds an existing numpy at install time. torchvision's
    # ToTensor hard-requires numpy; torch 2.2 wheels don't always drag it in.
    .pip_install("numpy<2", force_build=True)
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "pillow>=10.0",
        "fastapi[standard]>=0.110",
        force_build=True,
    )
    .run_function(_prefetch_dinov2, force_build=True)
)


ckpt_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Architecture mirrors species_identification/species_identification.ipynb.
_EMBED_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@app.cls(
    image=image,
    volumes={"/ckpt": ckpt_volume},
    gpu=None,             # change to "T4" (or similar) for GPU inference
    scaledown_window=300, # keep warm 5 min after last request
)
class Classifier:
    @modal.enter()
    def load(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision import transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("/ckpt/best_model.pth", map_location=device, weights_only=False)
        model_name = ckpt["model_name"]
        class_names = ckpt["class_names"]
        image_size = ckpt["image_size"]
        embed_dim = _EMBED_DIMS[model_name]

        backbone = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, len(class_names)),
                )

            def forward(self, x):  # type: ignore[override]
                with torch.no_grad():
                    return self.head(self.backbone(x))

        net = _Net().to(device).eval()
        net.load_state_dict(ckpt["model_state_dict"])

        self.model = net
        self.model_name = model_name
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        print(f"[modal] loaded {model_name} — {len(class_names)} classes — device={device}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def predict(self, request: Request, top_k: int = 3) -> dict:
        """Top-k classification of a raw image posted as the request body."""
        import torch
        from PIL import Image, UnidentifiedImageError

        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty body — POST raw image bytes.")
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
        except UnidentifiedImageError as e:
            raise HTTPException(status_code=400, detail=f"Not a decodable image: {e}")

        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
        probs = torch.softmax(logits.float(), dim=1).squeeze(0)
        k = max(1, min(top_k, probs.numel()))
        top_probs, top_idx = probs.topk(k)

        return {
            "model": self.model_name,
            "predictions": [
                {
                    "scientific_name": self.class_names[int(i)].replace("_", " "),
                    "probability": float(p),
                }
                for p, i in zip(top_probs.tolist(), top_idx.tolist())
            ],
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Cheap health check — also warms the container for the next POST."""
        return {
            "status": "ok",
            "model": self.model_name,
            "num_classes": len(self.class_names),
            "device": str(self.device),
        }
