"""
The one TFLite species classifier used on the device, driven by a manifest.

Why this exists
---------------
Every model we ship expects a different input domain, and nothing in a
.tflite file says which:

    MobileNetV3 (Keras, include_preprocessing=True)  raw pixels 0..255, uint8 I/O;
                                                     the [-1,1] rescale is IN the graph
    MobileNetV2 QAT (Keras, preprocessing outside)   pixels scaled to [-1,1], int8 I/O
    DINOv2 via ONNX -> onnx2tf                        ImageNet mean/std, int8 I/O

live_inference_mv2.py and live_inference_mv3.py each carried their own
`TFLiteClassifierTTA` with a hard-coded version of one of these, and the
temperature / class list came from whatever training run last wrote to the
shared outputs/ folder. Feeding a model the wrong domain doesn't crash — it
silently produces confident garbage — so the domain now travels with the
model in `model_manifest.json` and this class refuses to run without it.

Manifest (model_manifest.json, next to the .tflite)
---------------------------------------------------
    model_file        "model_int8.tflite" (relative to the manifest)
    arch              e.g. "mobilenetv3l"
    input_range       "raw_0_255" | "minus1_1" | "imagenet"
    img_size          320
    input_dtype       "uint8" | "int8" | "float32" (checked against the tensor)
    class_names       list, index order of the output logits
    negative_classes  ["seashore"] — never announced as a sighting
    temperature       T applied to logits before softmax (fit on int8+TTA val)
    id_threshold      min TTA-averaged confidence to accept an identification
    tta_views         [[crop_frac, hflip], ...]
    source / eval     provenance + the numbers behind temperature/threshold

Preprocessing per view: center crop -> optional h-flip -> resize to img_size
(INTER_AREA) -> normalize per input_range -> quantize with the input tensor's
scale/zero_point (rounded, clipped to the dtype range).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

MANIFEST_NAME = "model_manifest.json"
INPUT_RANGES = ("raw_0_255", "minus1_1", "imagenet")
DEFAULT_TTA_VIEWS = [[1.00, False], [1.00, True], [0.90, False], [0.80, False]]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DTYPES = {"uint8": np.uint8, "int8": np.int8, "float32": np.float32}


class ModelContractError(RuntimeError):
    """Model file and manifest disagree, or the manifest is missing/invalid."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ModelManifest:
    model_file: str
    arch: str
    input_range: str
    img_size: int
    input_dtype: str
    class_names: list[str]
    negative_classes: list[str] = field(default_factory=lambda: ["seashore"])
    temperature: float = 1.0
    id_threshold: float = 0.55
    tta_views: list = field(default_factory=lambda: [list(v) for v in
                                                     DEFAULT_TTA_VIEWS])
    source: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    created_at: str = ""
    # Set by load(); not serialized.
    path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.input_range not in INPUT_RANGES:
            raise ModelContractError(
                f"input_range={self.input_range!r} not in {INPUT_RANGES}")
        if self.input_dtype not in _DTYPES:
            raise ModelContractError(
                f"input_dtype={self.input_dtype!r} not in {tuple(_DTYPES)}")
        if not self.class_names:
            raise ModelContractError("manifest has no class_names")
        if float(self.temperature) <= 0:
            raise ModelContractError(f"temperature must be > 0, "
                                     f"got {self.temperature}")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds")

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ModelManifest":
        """Load from a manifest file or from the directory containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        if not path.is_file():
            raise ModelContractError(
                f"no {MANIFEST_NAME} at {path}. Every on-device model needs "
                f"a manifest declaring its input_range — export with "
                f"species_identification/export_tflite.py, or write one "
                f"explicitly.")
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f != "path"}
        unknown = set(data) - known
        if unknown:
            raise ModelContractError(f"unknown manifest keys in {path}: "
                                     f"{sorted(unknown)}")
        manifest = cls(**data)
        manifest.path = path
        return manifest

    @property
    def model_path(self) -> Path:
        base = self.path.parent if self.path else Path.cwd()
        return (base / self.model_file).resolve()

    def save(self, path: str | os.PathLike | None = None) -> Path:
        path = Path(path) if path else self.path
        if path is None:
            raise ValueError("no path to save manifest to")
        if path.is_dir():
            path = path / MANIFEST_NAME
        data = asdict(self)
        data.pop("path")
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.path = path
        return path


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Crop the central fraction of an (H, W, C) image."""
    if frac >= 1.0:
        return img
    h, w = img.shape[:2]
    new_h, new_w = int(h * frac), int(w * frac)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


def normalize(rgb: np.ndarray, input_range: str) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 in the model's training domain."""
    x = rgb.astype(np.float32)
    if input_range == "raw_0_255":
        return x
    if input_range == "minus1_1":
        return x / 127.5 - 1.0
    if input_range == "imagenet":
        return (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    raise ModelContractError(f"unknown input_range {input_range!r}")


def quantize(x: np.ndarray, scale: float, zero_point: int,
             dtype) -> np.ndarray:
    """Float tensor -> integer input tensor using the interpreter's params."""
    info = np.iinfo(dtype)
    q = np.round(x / scale + zero_point)
    return np.clip(q, info.min, info.max).astype(dtype)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def tta_probs(view_logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Average temperature-scaled softmax over views. view_logits is
    (views, classes) or (images, views, classes). This is exactly what the
    device computes, so eval can refit T on cached logits.
    """
    return softmax(view_logits / float(temperature), axis=-1).mean(axis=-2)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def build_interpreter(model_path: str | os.PathLike, num_threads: int = 4,
                      reference_kernels: bool = True):
    """
    Build a LiteRT interpreter. reference_kernels=True matches the UNO Q
    setup (XNNPACK disabled, BUILTIN_REF resolver) because XNNPACK has been
    flaky there with INT8 models; eval on a laptop can pass False for speed.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        OpResolverType = None

    if reference_kernels:
        os.environ["TFLITE_DISABLE_XNNPACK"] = "1"
        if OpResolverType is not None:
            try:
                return Interpreter(
                    model_path=str(model_path), num_threads=num_threads,
                    experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
            except (AttributeError, TypeError):
                pass
    return Interpreter(model_path=str(model_path), num_threads=num_threads)


@dataclass
class Identification:
    """Result of a one-shot identification with the acceptance policy applied."""
    label: str | None          # accepted species_id, or None
    top_label: str             # raw top-1 (may be a negative class)
    confidence: float          # TTA-averaged, temperature-scaled top-1 prob
    top: list[tuple[str, float]]
    reason: str                # "accepted" | "negative_class" | "low_confidence"

    @property
    def accepted(self) -> bool:
        return self.label is not None


class TFLiteClassifierTTA:
    """
    Manifest-driven TFLite classifier with test-time augmentation.

        clf = TFLiteClassifierTTA.from_dir("species_identification/outputs/deploy")
        preds = clf.predict(bgr_frame)          # [(label, prob), ...] top-5
        ident = clf.identify(bgr_frame)         # threshold + negative classes
    """

    def __init__(self, manifest: ModelManifest, *, num_threads: int = 4,
                 reference_kernels: bool = True) -> None:
        self.manifest = manifest
        model_path = manifest.model_path
        if not model_path.is_file():
            raise ModelContractError(f"model file not found: {model_path}")

        self.interpreter = build_interpreter(
            model_path, num_threads=num_threads,
            reference_kernels=reference_kernels)
        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.in_det["quantization"]
        self.out_scale, self.out_zp = self.out_det["quantization"]
        self.in_dtype = np.dtype(self.in_det["dtype"])
        self.out_dtype = np.dtype(self.out_det["dtype"])

        self.class_names = list(manifest.class_names)
        self.negative_classes = set(manifest.negative_classes)
        self.temperature = float(manifest.temperature)
        self.id_threshold = float(manifest.id_threshold)
        self.tta_views = [(float(c), bool(f)) for c, f in manifest.tta_views]
        self._validate()
        _, self.img_h, self.img_w, _ = (int(d) for d in self.in_det["shape"])

    @classmethod
    def from_dir(cls, model_dir: str | os.PathLike, **kwargs
                 ) -> "TFLiteClassifierTTA":
        return cls(ModelManifest.load(model_dir), **kwargs)

    def _validate(self) -> None:
        m = self.manifest
        shape = tuple(int(d) for d in self.in_det["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ModelContractError(
                f"expected NHWC [1,H,W,3] input, model has {shape}")
        if shape[1:3] != (m.img_size, m.img_size):
            raise ModelContractError(
                f"manifest img_size={m.img_size} but model input is "
                f"{shape[1]}x{shape[2]}")
        if self.in_dtype != np.dtype(_DTYPES[m.input_dtype]):
            raise ModelContractError(
                f"manifest input_dtype={m.input_dtype} but model input is "
                f"{self.in_dtype.name}")
        if self.in_dtype.kind in "iu" and not self.in_scale:
            raise ModelContractError("integer input tensor has no "
                                     "quantization scale")
        n_out = int(self.out_det["shape"][-1])
        if n_out != len(self.class_names):
            raise ModelContractError(
                f"model outputs {n_out} logits but manifest lists "
                f"{len(self.class_names)} class_names")
        unknown_neg = self.negative_classes - set(self.class_names)
        if unknown_neg:
            raise ModelContractError(f"negative_classes not in class_names: "
                                     f"{sorted(unknown_neg)}")

    # -- preprocessing -----------------------------------------------------

    def preprocess_float(self, rgb: np.ndarray, crop_frac: float = 1.0,
                         flip: bool = False) -> np.ndarray:
        """One TTA view in the model's float domain, (H, W, 3), unbatched."""
        img = center_crop(rgb, crop_frac)
        if flip:
            img = img[:, ::-1, :]
        img = cv2.resize(img, (self.img_w, self.img_h),
                         interpolation=cv2.INTER_AREA)
        return normalize(img, self.manifest.input_range)

    def preprocess_view(self, rgb: np.ndarray, crop_frac: float = 1.0,
                        flip: bool = False) -> np.ndarray:
        """One TTA view of an RGB uint8 image, batched and ready to invoke."""
        x = self.preprocess_float(rgb, crop_frac, flip)
        if self.in_dtype.kind in "iu":
            x = quantize(x, self.in_scale, self.in_zp, self.in_dtype.type)
        return np.expand_dims(x, 0)

    # -- inference -------------------------------------------------------------

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Invoke once; return float logits (dequantized if needed)."""
        self.interpreter.set_tensor(self.in_det["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_det["index"])[0]
        if self.out_dtype.kind in "iu":
            return (out.astype(np.float32) - self.out_zp) * self.out_scale
        return out.astype(np.float32)

    def view_logits(self, bgr: np.ndarray, tta: bool = True) -> np.ndarray:
        """(views, classes) logits for a BGR frame (as OpenCV delivers it)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        views = self.tta_views if tta else self.tta_views[:1]
        return np.stack([self.logits(self.preprocess_view(rgb, c, f))
                         for c, f in views])

    def predict(self, bgr: np.ndarray, top_k: int = 5,
                tta: bool = True) -> list[tuple[str, float]]:
        """Top-k (label, probability), TTA-averaged and temperature-scaled."""
        probs = tta_probs(self.view_logits(bgr, tta=tta), self.temperature)
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(self.class_names[i], float(probs[i])) for i in top_idx]

    def identify(self, bgr: np.ndarray, threshold: float | None = None
                 ) -> Identification:
        """
        One-shot identification with the manifest's acceptance policy:
        a negative class (e.g. seashore) or top-1 below id_threshold is
        refused rather than announced.
        """
        preds = self.predict(bgr)
        top_label, conf = preds[0]
        thr = self.id_threshold if threshold is None else threshold
        if top_label in self.negative_classes:
            return Identification(None, top_label, conf, preds,
                                  "negative_class")
        if conf < thr:
            return Identification(None, top_label, conf, preds,
                                  "low_confidence")
        return Identification(top_label, top_label, conf, preds, "accepted")

    def describe(self) -> str:
        m = self.manifest
        return (f"{m.arch} {self.img_w}x{self.img_h} "
                f"in={self.in_dtype.name}/{m.input_range} "
                f"out={self.out_dtype.name} classes={len(self.class_names)} "
                f"T={self.temperature:.3f} thr={self.id_threshold:.2f} "
                f"views={len(self.tta_views)}")
