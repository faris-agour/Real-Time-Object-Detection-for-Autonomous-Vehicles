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

### Phase 2: Object Detection Model Development
**Tasks**:
- Selected YOLOv8 architecture from Ultralytics for its balance between speed and accuracy.
- Trained and fine-tuned the YOLOv8 model on the KITTI dataset with enhanced regularization techniques and optimized augmentations.
- Exported the trained model to TensorRT format for accelerated inference.
- Evaluated the model based on key metrics: mAP (Mean Average Precision), IoU (Intersection over Union), and FPS (Frames Per Second).

**Deliverables**:
- YOLOv8 Training and Inference Script.
- TensorRT Exported Model (.engine file).
- Model Evaluation Report.

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
