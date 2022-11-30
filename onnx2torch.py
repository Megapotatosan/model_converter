import torch
import onnx2torch
import argparse
# Path to ONNX model
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-o", "--output", required=True, help="path to output model")

args = vars(ap.parse_args())
# You can pass the path to the onnx model to convert it or...
torch_model = onnx2torch.convert(args["model"])

print("[INFO] saving model...")
torch_model.save(args["output"], save_format="h5")
print('Output saved to: "{}"'.format(args["output"]))