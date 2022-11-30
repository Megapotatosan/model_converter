import tensorflow as tf
from keras.models import load_model
import tf2onnx
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])

spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)    #spec for tensorflow
output_path = model.name + ".onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)