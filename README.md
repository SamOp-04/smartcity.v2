# AI-Based Traffic Signal Optimization System

An intelligent traffic signal control system that uses computer vision, machine learning, and traffic simulation to optimize signal timing based on real-time vehicle detection from video feeds.

## Key Features

- **Real-time Vehicle Detection**: YOLOv8n-based detection with centroid tracking
- **Intelligent Signal Prediction**: XGBoost classifier (30/60/90/120s green time)
- **Multi-class Recognition**: Cars, buses/trucks, and motorcycles/bicycles
- **Weather-aware**: Adjusts recommendations based on rain conditions
- **SUMO Validation**: ~50% reduction in wait times vs fixed-time control
- **Interactive Dashboard**: Streamlit web interface for live monitoring

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAFFIC VIDEO INPUT                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 1: VEHICLE DETECTION (layer1_yolo/detector.py)           │
│  • YOLOv8n vehicle detection from video frames                  │
│  • Centroid-based tracking (prevents double-counting)           │
│  • Vehicle classification: Cars, Buses/Trucks, Bikes            │
│  • Output: Per-frame logs + 1-second window aggregates          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 2: ML PREDICTION (layer2_ml/)                            │
│  • XGBoost classifier with 4 discrete outputs                   │
│  • Input: vehicle counts + rain condition                       │
│  • Output: Signal timing (30s, 60s, 90s, 120s) + confidence     │
│  • Model accuracy: 91.5%                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 3: SUMO SIMULATION (layer3_sumo/)                        │
│  • Traffic simulation validation                                │
│  • Adaptive vs. fixed timing comparison                         │
│  • Performance metrics: wait times, throughput, queue length    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 4: WEB DASHBOARD (layer4_dashboard/)                     │
│  • Live video feed with detection overlays                      │
│  • Real-time vehicle count visualization                        │
│  • Signal timing recommendations                                │
│  • Simulation performance comparison                            │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
smartcity.v2/
├── traffic_pipeline.py      # Main orchestration script
├── test_pipeline.py         # Test suite
├── requirements.txt         # Pinned dependencies
├── yolov8n.pt              # YOLOv8 nano model
│
├── layer1_yolo/            # Vehicle Detection
│   ├── detector.py         # YOLO detection + centroid tracking
│   └── sample_images/      # Test images
│
├── layer2_ml/              # ML Prediction
│   ├── train_model.py      # XGBoost classifier training
│   ├── predict.py          # Inference module
│   ├── generate_dataset.py # Synthetic data generator
│   ├── data/               # Training data
│   └── models/             # Saved models (.joblib)
│
├── layer3_sumo/            # Traffic Simulation
│   ├── run_adaptive.py     # ML-based signal control
│   ├── run_fixed.py        # Fixed-time baseline
│   ├── compare.py          # A/B comparison
│   ├── config/             # SUMO configuration
│   ├── net/                # Network definition
│   └── routes/             # Traffic routes
│
├── layer4_dashboard/       # Web Interface
│   └── app.py              # Streamlit dashboard
│
└── utils/                  # Shared Utilities
    ├── constants.py        # Configuration constants
    ├── logging_config.py   # Logging setup
    └── validation.py       # Input validation
```

## Installation

### Prerequisites

- Python 3.9+
- SUMO (Simulation of Urban Mobility) - [Installation Guide](https://sumo.dlr.de/docs/Installing.html)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd smartcity.v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install SUMO (for Layer 3)

- **Windows**: Download from [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php)
- **Linux**: `sudo apt-get install sumo sumo-tools sumo-doc`
- **macOS**: `brew install sumo`

Verify: `sumo --version`

## Quick Start

### 1. Run the Dashboard

```bash
streamlit run layer4_dashboard/app.py
```

Open http://localhost:8501 - Features:
- **Tab 1**: Upload & process videos with live YOLO detection
- **Tab 2**: Predict green time from vehicle counts
- **Tab 3**: View/run SUMO simulation comparison

### 2. Process a Video (CLI)

```bash
python traffic_pipeline.py path/to/video.mp4
# With annotated output
python traffic_pipeline.py traffic.mp4 output.mp4
# With rain condition
python traffic_pipeline.py traffic.mp4 output.mp4 1
```

### 3. Run SUMO Comparison

```bash
python -m layer3_sumo.compare
# With GUI visualization
python -m layer3_sumo.compare --gui
```

### 4. Test the System

```bash
python test_pipeline.py
```

## Usage Examples

### Layer 1: Vehicle Detection

```python
from layer1_yolo.detector import detect_vehicles, process_video, process_video_live

# Single image detection
result = detect_vehicles("image.jpg")
print(f"Cars: {result['car_count']}, Buses: {result['bus_truck_count']}")

# Video processing (batch)
result = process_video("traffic.mp4", output_path="annotated.mp4")
print(f"Unique vehicles: {result['unique_totals']}")

# Live streaming (generator)
for data in process_video_live("traffic.mp4", frame_skip=5, max_frames=300):
    print(f"Frame {data['frame_number']}: {data['current']}")
    if data['done']:
        final = data
        break
```

**Vehicle Classification:**
| Group | COCO Classes |
|-------|-------------|
| CAR | 2 (car) |
| BUS_TRUCK | 5 (bus), 7 (truck) |
| BIKE | 1 (bicycle), 3 (motorcycle) |

### Layer 2: ML Prediction

```python
from layer2_ml.predict import predict_green_time, predict_green_time_class

# Simple prediction
green_time = predict_green_time(car_count=10, bus_truck_count=3, bike_count=5, rain=0)
print(f"Recommended: {green_time}s")

# Detailed prediction with confidence
result = predict_green_time_class(10, 3, 5, rain=0)
print(f"Time: {result['predicted_green_time']}s")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Probabilities: {result['probabilities']}")
```

**Output Classes:**
- **Class 0**: 30s (light traffic)
- **Class 1**: 60s (moderate traffic)
- **Class 2**: 90s (heavy traffic)
- **Class 3**: 120s (very heavy traffic)

### Layer 3: SUMO Simulation

```python
from layer3_sumo.run_adaptive import run_adaptive_simulation
from layer3_sumo.compare import compare

# Run adaptive simulation
results = run_adaptive_simulation(rain=0, sim_duration=600, gui=False)

# Compare both modes
comparison = compare(rain=0)
print(f"Wait time improvement: {comparison['improvement_wait_pct']}%")
print(f"Queue improvement: {comparison['improvement_queue_pct']}%")
```

## Configuration

### Video Processing (utils/constants.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_CONFIDENCE_THRESHOLD` | 0.5 | YOLO detection confidence |
| `DEFAULT_FRAME_SKIP` | 5 | Process every Nth frame |
| `DEFAULT_MAX_FRAMES` | 300 | Maximum frames to process |
| `DEFAULT_MAX_DIST` | 80 | Max distance for track matching (px) |
| `DEFAULT_MAX_FRAMES_MISSING` | 5 | Frames before track is dropped |

### ML Model Classes

```python
GREEN_TIME_CLASSES = [30, 60, 90, 120]
CLASS_BOUNDARIES = {
    30:  (0,   45),      # Light traffic
    60:  (45,  75),      # Moderate
    90:  (75,  105),     # Heavy
    120: (105, float("inf"))  # Very heavy
}
```

## Model Performance

### XGBoost Classifier
- **Accuracy**: 91.5%
- **Precision**: 91.45%
- **Recall**: 91.50%
- **F1-Score**: 91.42%

### SUMO Validation Results (600s simulation)

| Metric | Fixed | Adaptive | Improvement |
|--------|-------|----------|-------------|
| Avg Waiting Time | ~94s | ~47s | **~50%** |
| Avg Queue Length | ~6.6 | ~4.8 | **~28%** |
| Throughput | 272 | 279 | +7 vehicles |

## Retraining the Model

```bash
# Generate synthetic training data
python -m layer2_ml.generate_dataset

# Train XGBoost model
python -m layer2_ml.train_model

# Test predictions
python -m layer2_ml.predict
```

## Troubleshooting

### Model always predicts 30s
**Cause**: Training data imbalance (Class 0 dominates)
**Solution**: Generate balanced synthetic data with more high-traffic samples

### Vehicle counts seem wrong
**Cause**: Tracker threshold mismatch
**Solution**: Adjust `max_dist` (higher=fewer duplicates) or `frame_skip` (lower=more accuracy)

### Video processing is slow
**Solution**: Increase `frame_skip`, reduce resolution, or skip output video

### SUMO not working
**Cause**: SUMO not installed or not in PATH
**Solution**: Install SUMO and verify with `sumo --version`

## Technical Stack

| Component | Technology |
|-----------|------------|
| Vehicle Detection | YOLOv8n (Ultralytics) |
| Object Tracking | Centroid-based tracker |
| ML Model | XGBoost Classifier |
| Traffic Simulation | SUMO |
| Web Dashboard | Streamlit |
| Visualization | Plotly |
| Video Processing | OpenCV |

## Future Enhancements

- [ ] Real-time camera feed integration
- [ ] Multi-intersection coordination
- [ ] Emergency vehicle priority
- [ ] Historical pattern analysis
- [ ] Cloud deployment
- [ ] Mobile app

## License

MIT License

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
