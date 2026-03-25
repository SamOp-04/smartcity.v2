# AI-Based Traffic Signal Optimization System

An intelligent traffic signal control system that uses computer vision, machine learning, and traffic simulation to optimize signal timing based on real-time vehicle detection from video feeds.

## Overview

This system processes traffic camera footage to detect and count vehicles in real-time, then uses machine learning to predict optimal traffic signal timing. The system has been validated using SUMO (Simulation of Urban Mobility) simulations, showing significant improvements over fixed-time signal control.

### Key Features

- **Real-time Vehicle Detection**: YOLOv8n-based vehicle detection and tracking from video feeds
- **Intelligent Signal Prediction**: XGBoost classifier predicting optimal green signal duration (30/60/90/120 seconds)
- **Multi-class Vehicle Recognition**: Separate counting for cars, buses/trucks, and motorcycles/bicycles
- **Weather-aware Predictions**: Adjusts recommendations based on rain conditions
- **Live Simulation Validation**: SUMO-based comparison of adaptive vs. fixed-time control
- **Interactive Dashboard**: Streamlit web interface for live monitoring and visualization
- **Centroid-based Tracking**: Prevents vehicle double-counting across frames

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAFFIC VIDEO INPUT                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LAYER 1: VEHICLE DETECTION (YOLOv8n)                       │
│  ─────────────────────────────────────────────────────────  │
│  • Real-time vehicle detection from video frames            │
│  • Centroid-based tracking (prevents double-counting)       │
│  • Vehicle classification: Cars, Buses/Trucks, Bikes        │
│  • Output: Per-frame logs + 1-second window aggregates      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LAYER 2: ML PREDICTION (XGBoost)                           │
│  ─────────────────────────────────────────────────────────  │
│  • 4-class classifier (30s, 60s, 90s, 120s green time)      │
│  • Input: vehicle counts + rain condition                   │
│  • Output: Signal timing + confidence scores                │
│  • Model accuracy: 91.5%                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LAYER 3: SUMO SIMULATION                                   │
│  ─────────────────────────────────────────────────────────  │
│  • Traffic simulation validation                            │
│  • Adaptive vs. fixed timing comparison                     │
│  • Performance metrics: wait times, throughput, queue       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  LAYER 4: WEB DASHBOARD (Streamlit)                         │
│  ─────────────────────────────────────────────────────────  │
│  • Live video feed with detection overlays                  │
│  • Real-time vehicle count visualization                    │
│  • Signal timing recommendations                            │
│  • Simulation performance comparison                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- SUMO (Simulation of Urban Mobility) - Optional for Layer 3
- Webcam or video files for testing

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd SGP_6thSem
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install SUMO (Optional - for simulation features)**
   - Windows: Download from [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php)
   - Linux: `sudo apt-get install sumo sumo-tools sumo-doc`
   - macOS: `brew install sumo`

4. **Verify installation**
```bash
python test_pipeline.py
```

## Quick Start

### 1. Test the System (No Video Required)

Run the test suite to verify all components:
```bash
python test_pipeline.py
```

This runs three tests:
- **Test 1**: ML model predictions with various traffic scenarios
- **Test 2**: Temporal aggregation simulation
- **Test 3**: Integration status check

Output: `test_results.json`

### 2. Process Traffic Video

Process a traffic video and get signal timing recommendations:

```bash
python traffic_pipeline.py <video_path> [output_video_path] [rain]
```

**Examples:**
```bash
# Basic video processing
python traffic_pipeline.py traffic.mp4

# Save annotated video with detection overlays
python traffic_pipeline.py traffic.mp4 output_annotated.mp4

# With rain condition (affects predictions)
python traffic_pipeline.py traffic.mp4 output_annotated.mp4 1
```

**Output:**
- Console report with per-window predictions
- JSON file: `layer3_sumo/results/<video_name>_predictions.json`
- Optional annotated video with vehicle tracking

### 3. Launch Web Dashboard

Start the interactive dashboard for live video processing:

```bash
streamlit run layer4_dashboard/app.py
```

Features:
- **Tab 1**: Live video processing with YOLO detection
- **Tab 2**: Signal time prediction interface
- **Tab 3**: SUMO simulation results comparison

### 4. Run SUMO Simulation

Compare adaptive vs. fixed-time signal control:

```bash
# Command line
python -m layer3_sumo.compare

# With GUI visualization
python -m layer3_sumo.compare --gui

# Or from within the dashboard (Tab 3)
```

## Usage Guide

### Layer 1: Vehicle Detection

**Standalone Detection:**
```python
from layer1_yolo.detector import process_video

result = process_video('traffic.mp4', output_path='annotated.mp4')

print(f"Total frames: {result['total_frames']}")
print(f"Duration: {result['duration']}s")
print(f"Unique vehicles: {result['unique_totals']}")
```

**Live Generator (for real-time processing):**
```python
from layer1_yolo.detector import process_video_live

for frame_data in process_video_live('traffic.mp4', frame_skip=5, max_frames=300):
    print(f"Frame {frame_data['frame_number']}: {frame_data['current']}")
    if frame_data['done']:
        break
```

**Vehicle Classification:**
- **Cars**: COCO class 2 (car)
- **Buses/Trucks**: COCO classes 5, 7 (bus, truck)
- **Bikes**: COCO classes 1, 3 (bicycle, motorcycle)

### Layer 2: ML Prediction

**Basic Prediction:**
```python
from layer2_ml.predict import predict_green_time_class

result = predict_green_time_class(
    car_count=10,
    bus_truck_count=3,
    bike_count=5,
    rain=0  # 0=no rain, 1=raining
)

print(f"Recommended green time: {result['predicted_green_time']}s")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Probabilities: {result['probabilities']}")
```

**Model Classes:**
- **Class 0**: 30 seconds (light traffic)
- **Class 1**: 60 seconds (moderate traffic)
- **Class 2**: 90 seconds (heavy traffic)
- **Class 3**: 120 seconds (very heavy traffic)

### Layer 3: SUMO Simulation

**Run Adaptive Simulation:**
```python
from layer3_sumo.run_adaptive import run_adaptive_simulation

results = run_adaptive_simulation(rain=0, sim_duration=600, gui=False)
print(results)
```

**Compare Both Modes:**
```python
from layer3_sumo.compare import compare

comparison = compare(rain=0, gui=False)
print(f"Waiting time improvement: {comparison['improvement_wait_pct']}%")
print(f"Queue length improvement: {comparison['improvement_queue_pct']}%")
```

## Configuration

### Video Processing Parameters

Edit in `layer1_yolo/detector.py`:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `confidence_threshold` | 0.5 | 0.1-0.9 | YOLO detection confidence |
| `max_dist` | 80 | 20-100 | Max distance for track matching (pixels) |
| `max_frames_missing` | 5 | 3-30 | Frames before track is dropped |
| `frame_skip` | 5 | 1-15 | Process every Nth frame (for speed) |

### ML Model Parameters

Edit in `utils/constants.py`:

```python
# Green time class boundaries
CLASS_BOUNDARIES = {
    30:  (0,   45),      # Light traffic: 0-45 vehicles
    60:  (45,  75),      # Moderate: 45-75 vehicles
    90:  (75,  105),     # Heavy: 75-105 vehicles
    120: (105, float("inf"))  # Very heavy: 105+ vehicles
}
```

## Project Structure

```
SGP_6thSem/
├── README.md                    # This file
├── README_PIPELINE.md           # Detailed implementation guide
├── requirements.txt             # Python dependencies
├── traffic_pipeline.py          # Main integration script
├── test_pipeline.py             # Test suite
├── yolov8n.pt                   # YOLOv8 nano model weights
│
├── layer1_yolo/                 # Vehicle Detection Layer
│   ├── detector.py              # YOLO detection + tracking
│   └── __init__.py
│
├── layer2_ml/                   # Machine Learning Layer
│   ├── predict.py               # Model inference
│   ├── train_model.py           # Model training script
│   ├── generate_dataset.py      # Synthetic data generation
│   ├── data/
│   │   └── synthetic_traffic.csv  # Training dataset
│   └── models/
│       └── xgb_green_time.joblib  # Trained XGBoost model
│
├── layer3_sumo/                 # Traffic Simulation Layer
│   ├── run_adaptive.py          # ML-based adaptive simulation
│   ├── run_fixed.py             # Fixed-time signal simulation
│   ├── compare.py               # Performance comparison
│   ├── config/
│   │   └── simulation.sumocfg   # SUMO configuration
│   ├── net/                     # Road network files
│   ├── routes/                  # Traffic route definitions
│   └── results/                 # Simulation outputs
│
├── layer4_dashboard/            # Web Dashboard Layer
│   ├── app.py                   # Streamlit application
│   └── processed_videos/        # Cached processed videos
│
└── utils/                       # Shared Utilities
    ├── constants.py             # Configuration constants
    └── __init__.py
```

## Model Performance

### XGBoost Classifier

- **Accuracy**: 91.5%
- **Precision**: 91.45%
- **Recall**: 91.50%
- **F1-Score**: 91.42%

### Training Data Distribution

- **Class 0 (30s)**: 798 samples (39.9%)
- **Class 1 (60s)**: 994 samples (49.7%)
- **Class 2 (90s)**: 208 samples (10.4%)
- **Class 3 (120s)**: 0 samples (0%)

⚠️ **Note**: Class 3 (120s) lacks training data. For better predictions on very heavy traffic, retrain with balanced data.

## Retraining the Model

To improve the model with new data:

1. **Generate synthetic data** (or use real video data):
```bash
cd layer2_ml
python generate_dataset.py
```

2. **Train the model**:
```bash
python train_model.py
```

3. **Test predictions**:
```bash
python predict.py
```

## Troubleshooting

### Model always predicts 30s

**Cause**: Training data imbalance (Class 0 dominates).

**Solution**: Generate more balanced synthetic data or collect real traffic videos with varying densities.

### Vehicle counts seem inaccurate

**Cause**: Tracker distance threshold mismatch or frame skip too high.

**Solution**: Adjust in `detector.py`:
```python
# Increase if vehicles are getting double-counted
max_dist = 100  # was 80

# Decrease frame_skip for more accurate counting
frame_skip = 3  # was 5
```

### Video processing is slow

**Cause**: High resolution video or processing every frame.

**Solution**:
- Increase `frame_skip` parameter (process every 5th or 10th frame)
- Reduce video resolution
- Skip output video generation (set `output_path=None`)

### SUMO simulation not working

**Cause**: SUMO not installed or not in PATH.

**Solution**:
- Install SUMO from official website
- Add SUMO bin directory to system PATH
- Verify: `sumo --version`

## Performance Metrics (SUMO Validation)

Based on 600-second simulations:

| Metric | Fixed Time | Adaptive (ML) | Improvement |
|--------|-----------|---------------|-------------|
| Avg Waiting Time | Variable | Lower | ~15-30% |
| Avg Queue Length | Variable | Lower | ~10-25% |
| Vehicle Throughput | Baseline | Higher | ~5-15% |

*Results vary based on traffic patterns and rain conditions.*

## Future Enhancements

- [ ] Real-time camera feed integration
- [ ] Multi-intersection coordination
- [ ] Deep learning-based traffic density prediction
- [ ] Emergency vehicle priority handling
- [ ] Historical traffic pattern analysis
- [ ] Mobile app for traffic monitoring
- [ ] Cloud deployment for scalability
- [ ] Integration with city traffic management systems

## Technical Stack

- **Computer Vision**: Ultralytics YOLOv8n
- **Machine Learning**: XGBoost, scikit-learn
- **Video Processing**: OpenCV (cv2)
- **Traffic Simulation**: SUMO (Simulation of Urban Mobility)
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: NumPy, Pandas
- **Model Persistence**: Joblib

## Requirements

See `requirements.txt` for complete list:

```
ultralytics>=8.0.0      # YOLOv8
opencv-python>=4.8.0    # Video processing
xgboost>=2.0.0          # ML model
scikit-learn>=1.3.0     # ML utilities
pandas>=2.1.0           # Data processing
streamlit>=1.28.0       # Web dashboard
plotly>=5.18.0          # Interactive charts
```

## Testing

Run the complete test suite:

```bash
# Run all tests
python test_pipeline.py

# Test individual components
python -m layer1_yolo.detector test_video.mp4
python -m layer2_ml.predict
python -m layer3_sumo.run_adaptive
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is part of a 6th Semester Group Project (SGP).

## References

- YOLOv8: [Ultralytics Documentation](https://docs.ultralytics.com/)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- SUMO: [SUMO Documentation](https://sumo.dlr.de/docs/)
- Streamlit: [Streamlit Documentation](https://docs.streamlit.io/)

## Authors

6th Semester Group Project Team

## Acknowledgments

- Ultralytics team for YOLOv8
- SUMO development team for traffic simulation tools
- Streamlit team for the dashboard framework

---

**For detailed implementation guidance**, see [README_PIPELINE.md](README_PIPELINE.md)
