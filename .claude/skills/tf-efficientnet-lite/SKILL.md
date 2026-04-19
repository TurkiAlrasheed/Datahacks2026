---
name: tf-efficientnet-lite
description: "Use when picking, loading, preprocessing, or fine-tuning TF EfficientNet Lite (tf_efficientnet_lite0–4) via timm. Relevant whenever a lightweight / mobile / edge image classifier is needed — e.g. deploying the species classifier on the Arduino UNO Q instead of the DINOv2 ViT-B/14 backbone the notebooks currently use. Source: https://huggingface.co/docs/timm/models/tf-efficientnet-lite"
---

# TF EfficientNet Lite (timm)

Reference for the `tf_efficientnet_lite*` family of models inside `timm`. These are the mobile/edge variant of EfficientNet — Google's Tensorflow EfficientNet-Lite weights ported to PyTorch.

## When to use this skill

Invoke when the user:

- asks about `tf_efficientnet_lite0`, `tf_efficientnet_lite1`, `tf_efficientnet_lite2`, `tf_efficientnet_lite3`, `tf_efficientnet_lite4`, or "EfficientNet Lite" in general.
- wants a lightweight image classifier for on-device / edge / mobile inference.
- is considering replacing the project's current DINOv2 ViT-B/14 backbone with something small enough to run on the Arduino UNO Q SoC (quad-core ARM, no GPU). EfficientNet-Lite is a strong candidate and maps directly onto the existing `species_identification/species_identification.ipynb` training code (swap `torch.hub.load("facebookresearch/dinov2", ...)` for `timm.create_model("tf_efficientnet_lite0", ...)`).
- needs the image-preprocessing pipeline for an EfficientNet-Lite model (input size and normalization differ from DINOv2's).

Do **not** invoke for generic "EfficientNet" questions — use this skill only when the `lite` variant is specifically in play. For the full EfficientNet family, fall back to general timm knowledge.

## What it is

EfficientNet is a CNN family that uniformly scales depth / width / input resolution via a compound coefficient ϕ (Tan & Le, 2019). EfficientNet-Lite is the mobile-friendly derivative: it drops operations that are expensive or unsupported on constrained hardware.

Two concrete changes from stock EfficientNet:

- **ReLU6** activations replace Swish (friendlier to int8 quantization and to mobile inference runtimes).
- **Squeeze-and-Excitation blocks are removed** (SE uses dynamic global pooling that mobile accelerators handle poorly).

The backbone is still the inverted-bottleneck residual block from MobileNetV2. Weights in timm are ported from the [TensorFlow TPU repo](https://github.com/tensorflow/tpu).

## Variants

Five scaled variants are available in timm, ordered smallest → largest:

| timm name              | Rough use                                         |
|------------------------|---------------------------------------------------|
| `tf_efficientnet_lite0` | Smallest; first pick for Arduino UNO Q / phones. |
| `tf_efficientnet_lite1` | Small-medium.                                     |
| `tf_efficientnet_lite2` | Medium.                                           |
| `tf_efficientnet_lite3` | Medium-large.                                     |
| `tf_efficientnet_lite4` | Largest Lite variant; still edge-viable.          |

Always let `timm` + `resolve_data_config` supply the per-variant input size — do not hard-code 224 across the board.

## Load a pretrained model

```py
>>> import timm
>>> model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
>>> model.eval()
```

Replace the name with the variant of interest.

## Preprocessing: always use `resolve_data_config`

The input resolution, mean/std, and interpolation differ per variant. Let timm resolve them so training and inference agree:

```py
>>> import urllib
>>> from PIL import Image
>>> from timm.data import resolve_data_config
>>> from timm.data.transforms_factory import create_transform

>>> config = resolve_data_config({}, model=model)
>>> transform = create_transform(**config)

>>> url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
>>> urllib.request.urlretrieve(url, filename)
>>> img = Image.open(filename).convert('RGB')
>>> tensor = transform(img).unsqueeze(0) # transform and add batch dimension
```

## Inference

```py
>>> import torch
>>> with torch.inference_mode():
...     out = model(tensor)
>>> probabilities = torch.nn.functional.softmax(out[0], dim=0)
>>> print(probabilities.shape)
>>> # prints: torch.Size([1000])
```

## Top-k ImageNet labels

```py
>>> # Get imagenet class mappings
>>> url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
>>> urllib.request.urlretrieve(url, filename)
>>> with open("imagenet_classes.txt", "r") as f:
...     categories = [s.strip() for s in f.readlines()]

>>> # Print top categories per image
>>> top5_prob, top5_catid = torch.topk(probabilities, 5)
>>> for i in range(top5_prob.size(0)):
...     print(categories[top5_catid[i]], top5_prob[i].item())
>>> # prints class names and probabilities like:
>>> # [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
```

For feature extraction, follow timm's feature extraction docs (`forward_features`, `forward_head` with `pre_logits=True`, or `features_only=True` at `create_model` time).

## Fine-tuning

Swap the classifier head by passing `num_classes` at construction time — timm automatically rebuilds the final linear layer with the new shape while keeping the backbone weights:

```py
>>> model = timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
```

Then either write a standard PyTorch training loop or adapt timm's [`train.py`](https://github.com/rwightman/pytorch-image-models/blob/master/train.py).

### Notes for this project specifically

If you port `species_identification.ipynb` to EfficientNet-Lite:

- Replace the DINOv2 backbone + hand-written `_DINOv2Classifier` head with `timm.create_model('tf_efficientnet_lite0', pretrained=True, num_classes=len(class_names))` — timm already builds a head.
- Replace the ImageNet-mean normalization block in the transform pipeline with the output of `resolve_data_config({}, model=model)` / `create_transform(**config)`. The Lite variants use different input sizes (e.g. 224 for lite0, 300 for lite3).
- Keep the rest of the training scaffold (AdamW + cosine LR + AMP + early stopping, ImageFolder, stratified 70/15/15 split). These are orthogonal to the backbone.
- Unfreezing the whole Lite backbone is cheap (it's small), so `FREEZE_BACKBONE = False` with `LR_FINETUNE = 1e-4` is usually a better starting point than linear probing.
- The `seashore` negative class still lives in `class_names` and must survive the swap — restore it from the checkpoint, never re-derive from a directory listing.

## Training from scratch

Follow timm's [training recipe scripts](https://github.com/rwightman/pytorch-image-models) if you actually want to retrain rather than fine-tune.

## Citation

```BibTeX
@misc{tan2020efficientnet,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
