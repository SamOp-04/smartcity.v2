# AI-Based Traffic Signal Optimization System

An intelligent traffic management system that uses computer vision and machine learning to dynamically optimize traffic signal timing, reducing wait times and improving traffic flow.

## Architecture

```
Traffic Video Input
       |
[Layer 1: YOLO Detection] --> Vehicle counts (cars, buses, bikes)
       |
[Layer 2: ML Prediction]  --> Optimal green light duration (30/60/90/120s)
       |
[Layer 3: SUMO Simulation] --> Performance validation
       |
[Layer 4: Dashboard]      --> Real-time visualization
```

## Project Structure

```
smartcity.v2/
├── traffic_pipeline.py      # Main orchestration script
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
│   └── models/             # Saved models
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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Dashboard

```bash
streamlit run layer4_dashboard/app.py
```

Open http://localhost:8501 in your browser.

### 2. Process a Video

```bash
python -m layer1_yolo.detector path/to/video.mp4 output.mp4
```

### 3. Run Full Pipeline

```bash
python traffic_pipeline.py path/to/video.mp4
```

### 4. Run SUMO Simulation Comparison

```bash
python -m layer3_sumo.compare
# Add --gui flag for visual simulation
```

## Layer Details

### Layer 1: YOLO Vehicle Detection

Uses YOLOv8 nano model for real-time vehicle detection with centroid-based tracking.

**Features:**
- Detects cars, buses, trucks, motorcycles, and bicycles
- Centroid tracking prevents double-counting
- Configurable confidence threshold and frame skip
- Live streaming via generator pattern

**Vehicle Groups:**
| Group | COCO Classes |
|-------|-------------|
| CAR | car |
| BUS_TRUCK | bus, truck |
| BIKE | bicycle, motorcycle |

### Layer 2: ML Signal Time Prediction

XGBoost classifier predicts optimal green signal duration.

**Input Features:**
- `car_count` - Number of cars
- `bus_truck_count` - Number of buses/trucks
- `bike_count` - Number of bikes
- `rain` - Weather condition (0/1)

**Output Classes:** 30s, 60s, 90s, or 120s green time

**Train the model:**
```bash
python -m layer2_ml.generate_dataset  # Generate synthetic data
python -m layer2_ml.train_model       # Train XGBoost model
```

### Layer 3: SUMO Simulation

Validates ML predictions using traffic simulation.

**Simulation Setup:**
- 4-way intersection with traffic light
- 600-second simulation duration
- Multiple vehicle types and flows

**Results (typical):**
| Metric | Fixed | Adaptive | Improvement |
|--------|-------|----------|-------------|
| Avg Wait Time | ~94s | ~47s | ~50% |
| Avg Queue Length | ~6.6 | ~4.8 | ~28% |

### Layer 4: Streamlit Dashboard

Interactive web interface with three tabs:

1. **Vehicle Detection** - Upload videos, watch live YOLO detection
2. **Signal Prediction** - Predict green time from vehicle counts
3. **Simulation Results** - Compare fixed vs adaptive control

## API Usage

### Detect Vehicles

```python
from layer1_yolo.detector import detect_vehicles, process_video

# Single image
result = detect_vehicles("image.jpg")
print(f"Cars: {result['car_count']}, Buses: {result['bus_truck_count']}")

# Video processing
result = process_video("video.mp4", "output.mp4")
print(f"Unique vehicles: {result['unique_totals']}")
```

### Predict Green Time

```python
from layer2_ml.predict import predict_green_time, predict_green_time_class

# Simple prediction
green_time = predict_green_time(car_count=10, bus_truck_count=3, bike_count=5, rain=0)
print(f"Recommended: {green_time}s")

# Detailed prediction with confidence
result = predict_green_time_class(10, 3, 5, rain=0)
print(f"Time: {result['predicted_green_time']}s")
print(f"Confidence: {result['confidence']:.1%}")
```

### Run Simulation

```python
from layer3_sumo.compare import compare

results = compare(rain=0)
print(f"Wait time improvement: {results['improvement_wait_pct']}%")
```

## Configuration

Key constants are defined in `utils/constants.py`:

```python
GREEN_TIME_CLASSES = [30, 60, 90, 120]  # Possible green times
MIN_GREEN_TIME = 10                      # Minimum green time
MAX_GREEN_TIME = 120                     # Maximum green time
YELLOW_TIME = 3                          # Yellow light duration
DEFAULT_CONFIDENCE_THRESHOLD = 0.5       # YOLO detection threshold
```

## Performance

- **ML Model Accuracy:** ~91.5%
- **Detection Speed:** Real-time with frame skipping
- **Traffic Improvement:** ~50% reduction in average wait time

## Troubleshooting

### SUMO not found
Make sure SUMO is installed and `sumo` is in your PATH:
```bash
sumo --version
```

### Model not found
Train the ML model first:
```bash
python -m layer2_ml.train_model
```

### Video processing slow
Increase frame skip in the dashboard or use:
```python
process_video_live(video_path, frame_skip=10, max_frames=300)
```

## License

MIT License
