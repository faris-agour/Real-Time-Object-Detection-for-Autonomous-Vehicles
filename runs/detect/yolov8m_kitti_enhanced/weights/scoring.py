import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
import cv2

# Global variable to hold the ONNX model session
session = None


# This function is called when the service starts to initialize the model
def init():
    global session
    # Load the ONNX model using ONNX Runtime
    session = ort.InferenceSession("model.onnx")


# This function will be called for each inference request
def run(raw_data):
    try:
        # Preprocess input (decode base64 image data)
        img_data = base64.b64decode(raw_data)  # Decode the base64 image data
        img = Image.open(io.BytesIO(img_data))  # Open image from binary data
        img = img.resize((640, 640))  # Resize to YOLOv8 input size
        img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0  # Normalize and transpose for YOLOv8
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run inference with the model
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})

        # Post-process outputs (e.g., filter boxes, NMS)
        predictions = process_outputs(outputs)  # This is your custom function for output processing
        return json.dumps({"predictions": predictions})

    except Exception as e:
        return json.dumps({"error": str(e)})


# Your custom logic to process the raw outputs (e.g., apply NMS, thresholding)
def process_outputs(outputs):
    # Example: filter and process detections from YOLOv8
    boxes, confidences, class_ids = outputs[0], outputs[1], outputs[2]

    # Example post-processing logic (simplified)
    detections = []
    for i in range(len(boxes)):
        if confidences[i] > 0.5:  # Apply confidence threshold
            detection = {
                "class_id": int(class_ids[i]),
                "confidence": float(confidences[i]),
                "box": boxes[i].tolist()  # Convert bounding box to list
            }
            detections.append(detection)

    return detections
