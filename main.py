# Using FastAPI (recommended for better performance)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("runs/detect/yolov8m_kitti_enhanced/weights/best.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform detection
    detections = model.predict(image)

    return JSONResponse(detections)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}