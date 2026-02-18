# Eye-Tracking prototype
This repository contains a proof-of-concept eye-tracking algorithm focused on webcam-based pupil detection and tracking in video recordings. It processes eye-region videos to detect pupil contours and visualize gaze movements, and outputs a .csv file containing the eye-gaze coordinates and categorization (left/right).

## Project Structure

├── src/                    # Eye-tracking algorithm implementation
├── data/
│   ├── raw/               # Original eye movement video files
│   └── processed/         # Output videos with tracking overlays + CSV files
├── requirements.txt       # Python dependencies
└── README.md

## Key Features
- Binary gaze classification (left/center/right)
- Lightweight OpenCV-based processing
- Frame-by-frame CSV output
- Annotated video overlay for quality control

## Installation
`
pip install -r requirements.txt
`
## Pipeline
1. Eye region extraction from video frame
2. Grayscale conversion + Gaussian blur
3. Binary thresholding for pupil detection
4. Contour analysis (largest = pupil)
5. Binary gaze classification via horizontal position


