# Parking Spot Detection Algorithm

A computer vision-based system that automatically detects parking spots from aerial/overhead images and determines their occupancy status by comparing an empty parking lot reference image with a current image.

## 🚗 Features

- **Automatic Parking Spot Detection**: Identifies individual parking spaces from aerial images
- **Occupancy Status Detection**: Determines if spots are occupied or empty
- **Real-time Visual Feedback**: Shows detection process step-by-step with visual windows
- **JSON Output**: Exports results in structured format for further processing
- **Confidence Scoring**: Provides confidence metrics for each detection

## 🛠️ Technologies Used

- **OpenCV**: Primary image processing and computer vision library
- **NumPy**: Numerical computing for array operations
- **scikit-learn**: Machine learning library (DBSCAN clustering)
- **Python 3.x**: Core programming language

## 📋 Requirements

```bash
pip install opencv-python numpy scikit-learn
```

## 🚀 Quick Start

### Basic Usage

```python
from parking_detection import detect_parking_spots_with_occupancy

# Detect parking spots and occupancy
spots = detect_parking_spots_with_occupancy(
    empty_lot_path="empty_parking_lot.jpg",
    filled_lot_path="current_parking_lot.jpg",
    output_json_path="results.json",
    output_image_path="detected_spots.jpg"
)

# Print results
occupied = sum(1 for spot in spots if spot['occupied'])
total = len(spots)
print(f"Total spots: {total}")
print(f"Occupied: {occupied}")
print(f"Available: {total - occupied}")
```

### Input Requirements

1. **Empty Reference Image**: A photo of the parking lot when completely empty
2. **Current Image**: A photo of the parking lot at the time you want to check occupancy

Both images should be:
- Taken from the same angle/perspective
- Have clearly visible white parking line markings
- Be in a supported format (JPG, PNG, etc.)

## 🔧 Algorithm Pipeline

The algorithm works through several key steps:

### 1. Image Loading and Preprocessing
- Loads reference (empty) and current images
- Converts to appropriate color spaces for processing

### 2. White Line Detection
- Converts images to HSV color space for better color filtering
- Detects white parking lines using color thresholding
- Applies morphological operations to clean up detected lines

### 3. Line Segment Detection
- Uses Hough Line Transform to detect line segments
- Classifies lines as horizontal or vertical based on angle
- Parameters: `threshold=20`, `minLineLength=25`, `maxLineGap=10`

### 4. Line Merging and Grid Formation
- Uses DBSCAN clustering to group nearby parallel lines
- Extends line segments across the full image
- Creates a complete grid structure

### 5. Parking Spot Generation
- Generates rectangular parking spots from grid intersections
- Filters out invalid spots based on size constraints:
  - Width: 40-150 pixels
  - Height: 80-250 pixels

### 6. Occupancy Detection
- Compares empty reference with current image
- Uses Gaussian blur to reduce noise
- Calculates pixel difference percentage
- Classifies as occupied if change > 3% (configurable threshold)

## 📊 Output Format

### JSON Output
```json
{
    "timestamp": "1234567890",
    "total_spots": 45,
    "occupied_spots": 12,
    "empty_spots": 33,
    "spots": [
        {
            "id": 1,
            "rect": [100, 150, 80, 120],
            "occupied": true,
            "change_percentage": 15.7,
            "confidence": 0.8
        }
    ]
}
```

### Visual Output
The algorithm displays 5 processing windows:
1. **Empty Lot Reference**: Original empty parking lot image
2. **White Line Mask**: Detected white parking lines
3. **Raw Line Segments**: Individual line segments found
4. **Merged & Extended Grid**: Complete grid structure
5. **Spots with Occupancy Status**: Final results with colored rectangles

## ⚙️ Configuration

### Key Parameters You Can Adjust

```python
# White line detection thresholds
lower_white = np.array([0, 0, 200])    # HSV lower bound
upper_white = np.array([180, 25, 255]) # HSV upper bound

# Hough line detection
threshold = 20        # Minimum votes for line detection
minLineLength = 25    # Minimum line length in pixels
maxLineGap = 10       # Maximum gap between line segments

# DBSCAN clustering
eps = 15             # Maximum distance between points in cluster
min_samples = 1      # Minimum points needed to form cluster

# Spot size filtering
MIN_WIDTH, MAX_WIDTH = 40, 150      # Width constraints in pixels
MIN_HEIGHT, MAX_HEIGHT = 80, 250    # Height constraints in pixels

# Occupancy detection
threshold = 3        # Change percentage threshold for occupancy
```

## 🚀 Advanced Improvements

For enhanced performance, consider implementing:

- **Adaptive Thresholding**: Handle varying lighting conditions
- **Deep Learning Integration**: More robust detection using neural networks
- **Temporal Analysis**: Use multiple frames for better accuracy
- **Camera Calibration**: Correct for perspective distortions
- **Multi-scale Processing**: Handle different image resolutions

**Note**: This algorithm works best with parking lots that have clear, visible white line markings and consistent lighting conditions. For optimal results, ensure your reference (empty) image and current images are taken under similar lighting conditions and from the same perspective.
