# Real-Time Object Detection for Autonomous Vehicles

## Project Overview
The Real-Time Object Detection for Autonomous Vehicles project focuses on building a machine learning model capable of detecting and classifying objects in the environment (pedestrians, vehicles, traffic signs, obstacles) in real-time. This model will be deployed in autonomous vehicle systems to enhance safety and decision-making during driving.

The project addresses challenges such as detecting objects in varying environmental conditions (lighting, road types, and weather), ensuring safety in real-time driving scenarios, and integrating the model into an operational autonomous vehicle system.

## Key Features
✅ **Object Detection**: Detects and classifies objects such as pedestrians, vehicles, traffic signs, and obstacles in real-time.  
✅ **Real-Time Performance**: Optimized for fast, real-time inference using TensorRT.  
✅ **Environmental Adaptation**: Handles different driving environments including urban streets, highways, night conditions, and adverse weather.  
✅ **MLOps Pipeline**: Implements MLOps practices with MLflow to monitor and improve the model post-deployment.  

## Phases & Deliverables

### Phase 1: Data Collection, Exploration & Preprocessing
**Tasks**:
- Collect datasets such as KITTI.
- Explore and analyze dataset composition, image quality, and environmental factors.
- Preprocess data by resizing, normalizing, and augmenting images.

**Deliverables**:
- Dataset Exploration Report.
- Preprocessed and augmented dataset ready for training.

### Phase 2: Model Development and Model Evaluation
**Tasks**:
- Selected YOLOv8 architecture from Ultralytics for its balance between speed and accuracy.
- Trained and fine-tuned the YOLOv8 model on the KITTI dataset with enhanced regularization techniques and optimized augmentations.
- Exported the trained model to TensorRT format for accelerated inference.

**Deliverables**:
- YOLOv8 Training and Inference Script.
- TensorRT Exported Model (.engine file).
- Model Evaluation Report.
## 🚀 Model Performance Highlights

### 📊 Key Metrics (Best Epoch 131)
| Metric               | Score   | What It Means |
|----------------------|---------|---------------|
| **Precision**        | 0.899   | 90% of detected objects were correct (low false positives) |
| **Recall**          | 0.853   | Detected 85% of all actual objects (few misses) |
| **mAP@0.5**        | 0.906   | Excellent overall detection at standard IoU threshold |
| **mAP@0.5:0.95**  | 0.673   | Good performance across varying detection strictness |

### 🏆 Class Detection Superstars
| Class       | AP@0.5 | Performance |
|-------------|--------|-------------|
| 🚛 Truck    | 0.982  | Near-perfect |
| 🚋 Tram     | 0.987  | Best performer |
| 🚗 Car      | 0.972  | Exceptional |
| 🚐 Van      | 0.970  | Flawless |

### 💪 Strengths
- Dominates vehicle detection (cars, trucks, vans >96% accuracy)
- Low false positives (high precision)
- Excellent generalization (validation matches training)

### 🔧 Areas for Improvement
- Pedestrian detection (AP 0.760) - smaller objects need work
- Person sitting (AP 0.824) - pose variations challenging
- mAP@0.5:0.95 shows room for tighter bounding boxes

> ✅ **Conclusion**: Our YOLOv8 model achieves production-ready performance on critical vehicle detection while maintaining strong overall accuracy (90.6% mAP@0.5). Perfect balance for real-time autonomous vehicle needs.
> 
### Phase 3: Deployment & Real-Time Testing
**Tasks**:
- Deploy the model into an optimized inference pipeline using TensorRT.
- Integrate the model with vehicle camera inputs for real-time detection.
- Test the model in real-world driving scenarios and record performance.

**Deliverables**:
- Deployed TensorRT Model.
- Real-Time Testing Report.

### Phase 4: MLOps & Monitoring
**Tasks**:
- Set up an MLOps pipeline using MLflow for continuous monitoring and automated retraining.
- Track model performance, detect drift, and ensure system reliability.

**Deliverables**:
- MLOps Report.
- Monitoring Infrastructure Setup.

### Phase 5: Final Documentation & Presentation
**Tasks**:
- Summarize the project in a final report covering all stages.
- Prepare a comprehensive presentation for stakeholders.

**Deliverables**:
- Final Project Report.
- Final Presentation.

## Technologies Used
- **Object Detection Model**: YOLOv8 (Ultralytics)  
- **Frameworks**: PyTorch, Ultralytics YOLO library  
- **MLOps Tools**: MLflow  
- **Data Augmentation**: Ultralytics built-in augmentations  
- **Deployment & Real-Time Inference**: NVIDIA TensorRT, OpenCV  
- **Version Control**: GitHub  
