import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "en_US-danny-low.opt.onnx"
so.intra_op_num_threads = 4

# Creating the session writes the optimized model to disk as a side effect
_ = ort.InferenceSession(
    "en_US-danny-low.onnx",
    sess_options=so,
    providers=['CPUExecutionProvider']
)
print("Optimized model written to en_US-danny-low.opt.onnx")