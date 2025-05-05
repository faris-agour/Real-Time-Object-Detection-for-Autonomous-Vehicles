from ultralytics import YOLO

# Load medium-sized model for better generalization
model = YOLO("yolov8m.pt")  # Better balance than 'l' for KITTI

# Train the model with enhanced regularization
results = model.train(
    
    # Dataset configuration
    data="data/kitti_dataset/kitti.yaml",
    epochs=200,  # Increased for convergence
    imgsz=640,  # Higher resolution for better accuracy
    batch=16,  # Reduced for better gradient estimation
    
    # Optimized hyperparameters
    optimizer="AdamW",
    lr0=0.0005,
    cos_lr=True,
    momentum=0.9,
    weight_decay=0.01,  # Stronger regularization
    label_smoothing=0.1,  # Added regularization
    dropout=0.2,  # Add dropout layers
    
    # Enhanced augmentations
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.9,
    shear=0.2,
    perspective=0.0005,
    mosaic=1.0,
    mixup=0.05,  # Conservative mixup
    copy_paste=0.05,
    flipud=0.1,
    
    # Training setup
    device=0,
    workers=20,  # Match your CPU cores
    name="yolov8m_kitti_enhanced",
    patience=15,  # Earlier stopping
    deterministic=True,
    pretrained=True,
    erasing=0.1,
)

# Export to TensorRT with optimization
model.export(
    format="engine",
    imgsz=640,  # Match training size
    device=0,
    workspace=16,  # Use more if GPU memory allows
    int8=True,     # Enable INT8 quantization (optional)
    simplify=True,
    name="yolov8m_kitti_tensorrt",
)


# Example inference code
def run_inference(model_path, source_dir):
    model = YOLO(model_path)
    results = model.predict(
        source=source_dir,
        conf=0.3,  # Adjusted confidence threshold
        iou=0.5,  # Adjusted IoU threshold
        imgsz=640,  # Match training size!
        device=0,
        stream=True,  # For video processing
        augment=False,  # Disable TTA for speed
    )
    return results


# Usage
trt_model = "yolov8m_kitti_tensorrt.engine"
# Run inference on the KITTI dataset
run_inference(trt_model, "data/kitti_dataset/data_object_image_2/testing/image_2")