"""
Live camera loop for the MobileNetV2 QAT model.

Same loop as live_inference_mv3.py — this file only changes the default
--model-dir. The preprocessing that used to be hand-written here (scale to
[-1, 1], quantize to int8 with the tensor's scale/zero_point) is now declared
by the model's manifest (input_range "minus1_1", input_dtype "int8") and
applied by the shared classifier in
species_identification/vision/tflite_classifier.py.

History worth keeping: the first version of this script fed raw [0, 255]
pixels to the int8 input and only recognized uint8 inputs, so every frame
saturated and predictions were near-random — including on training images.
Declaring the input domain in the manifest is what prevents that class of
bug from coming back.

Run:
    python3 live_inference_mv2.py
    python3 live_inference_mv2.py --camera-index 0
"""

import sys
from pathlib import Path

from live_inference_mv3 import main

MV2_QAT_DIR = (Path(__file__).resolve().parent / "species_identification"
               / "outputs" / "mobilenetv2_qat")


if __name__ == "__main__":
    sys.exit(main(default_model_dir=MV2_QAT_DIR))
