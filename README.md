# Vision-I: Vision Enhancement System for Visually Impaired Users

Vision-I is an assistive technology system that uses computer vision and artificial intelligence to help visually impaired users perceive their surroundings. The system uses YOLOv8 object detection combined with text-to-speech feedback to identify objects, estimate distances, and even recognize U.S. currency denominations.

## Features

- **Real-time object detection** using YOLOv8
- **Distance estimation** based on object size in the frame
- **Proximity warnings** for close objects
- **U.S. currency recognition** to identify dollar bill denominations
- **Voice feedback** through text-to-speech
- **Simple keyboard controls** for easy operation

## Requirements

- Python 3.6 or higher
- Webcam or compatible camera
- Speaker for audio output
- CUDA-compatible GPU recommended for better performance (but not required)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/vision-i.git
   cd vision-i
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (will be downloaded automatically on first run)

## Usage

Run the Vision-I system with:

```bash
python vision_i_revised.py
```

### Controls

- **'d' key**: Detect and announce all objects in the current frame
- **'r' key**: Activate dollar bill detection mode
- **'q' key**: Quit the application

## How It Works

1. **Object Detection**: The system uses YOLOv8 to detect common objects in the camera's view.
2. **Distance Estimation**: Using the size of the bounding box, the system estimates the approximate distance to each object.
3. **Proximity Warning**: When objects are close to the center of the frame and within a certain distance, the system issues a warning.
4. **Dollar Bill Recognition**: Using computer vision techniques including contour detection, color analysis, and aspect ratio checking to identify and classify U.S. currency.
5. **Voice Feedback**: All detections and warnings are communicated through voice using the pyttsx3 text-to-speech engine.

## System Components

- **YOLOv8**: State-of-the-art object detection model
- **OpenCV**: Computer vision library for image processing
- **pyttsx3**: Text-to-speech engine for voice feedback
- **NumPy**: Numerical computing for data processing
- **PIL**: Python Imaging Library for image handling

## Limitations

- Currency detection works best with well-lit, unfolded U.S. dollar bills
- Distance estimation is approximate and relative, not measured in absolute units
- Object detection accuracy depends on lighting conditions and camera quality
- Currently only supports English language for voice feedback

## Future Improvements

- Add support for more currencies
- Implement optical character recognition (OCR) for reading text
- Add face recognition for identifying known people
- Improve distance estimation accuracy
- Add support for multiple languages
- Develop a mobile application version

## License

[MIT License](LICENSE)

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- All contributors and testers who helped improve this system

## Contact

For questions, suggestions, or contributions, please open an issue on GitHub or contact [your-email@example.com].
