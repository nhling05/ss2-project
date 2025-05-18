# Vision-I: Vision Enhancement System for Visually Impaired Users

Vision-I is an assistive technology system that uses computer vision and artificial intelligence to help visually impaired users perceive their surroundings. The system combines YOLOv8 object detection with text-to-speech feedback to identify objects, estimate distances, and recognize U.S. currency denominations.

## Key Features

- **Real-time object detection** using YOLOv8
- **Distance estimation** based on object size in the frame
- **Proximity warnings** for objects close to the center of the frame
- **U.S. currency recognition** to identify dollar bill denominations
- **Voice feedback** through text-to-speech engine (pyttsx3)
- **Simple keyboard controls** for easy operation

## System Requirements

- Python 3.6 or higher
- Webcam or compatible camera (DroidCam support included)
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
   
   Main dependencies include:
   - numpy
   - opencv-python
   - pyttsx3
   - pillow
   - matplotlib
   - ultralytics
   - torch
   - keyboard

3. Download the YOLOv8 model (will be downloaded automatically on first run)

## Usage

Run the Vision-I system with:

```bash
python vision_i_revised.py
```

### Keyboard Controls

- **'Space' key (single press)**: Toggle between object detection mode and currency detection mode
- **'Space' key (double press)**: Suppress/unsuppress warnings for the center object
- **'R' key**: Repeat all detected objects in the current frame
- **'ESC' key**: Exit the application

### Create Desktop Shortcut

To create a desktop shortcut, run:

```bash
python vision_i_revised.py --create-shortcut
```

## How It Works

### 1. Object Detection
The system uses YOLOv8 to detect common objects in the camera's view. The lightweight YOLOv8n model is used to ensure performance on standard hardware systems.

### 2. Distance Estimation
Using the size of the bounding box, the system estimates the relative distance to each object according to these levels:
- "very close": object occupies >60% of frame width
- "close": object occupies >40% of frame width
- "moderate distance": object occupies >25% of frame width
- "far": object occupies >15% of frame width
- "very far": object occupies <15% of frame width

### 3. Proximity Warning
When objects are close to the center of the frame (within the CENTER_ZONE) and at a close distance, the system issues a voice warning. Each object is warned about a maximum of 2 times to avoid excessive repetition.

### 4. Currency Recognition
The system uses two methods to recognize U.S. currency:
- **ML Model Method**: Uses a specialized YOLOv8 model trained for recognizing bill denominations ($1, $5, $10, $20, $50, $100) with both front and back sides.
- **Image Processing Method**: Uses shape analysis, aspect ratio, and color analysis to identify denominations when the ML model is unavailable.

### 5. Voice Feedback
All detections and warnings are communicated through voice using the pyttsx3 engine. The system automatically manages timing between announcements to avoid excessive repetition.

## Camera Connection

The system supports connection to:
- Standard computer webcams
- Android phones as webcams via DroidCam with various ports and URLs (192.168.0.113:4747, 127.0.0.1:4747, localhost:4747, and port 8080)

The connection process automatically tries multiple configurations to find a suitable camera.

## Current Limitations

- Currency recognition works best with well-lit, unfolded U.S. dollar bills
- Distance estimation is relative, not measured in absolute units
- Object detection accuracy depends on lighting conditions and camera quality
- Currently only supports English language for voice feedback
- Warnings may repeat if objects move and are considered new objects

## Future Improvements

- Add support for more currencies
- Implement optical character recognition (OCR) for reading text
- Add face recognition for identifying known people
- Improve distance estimation accuracy
- Add support for multiple languages
- Develop a mobile application version
- Optimize performance for low-end devices
- Improve object detection in varied lighting conditions

## Code Structure

- **vision_i_revised.py**: Main system file
- **droidcam_connector.py**: Module for connecting to DroidCam on Android devices
- **models/best_money.pt**: YOLOv8 model trained for currency recognition (needs to be downloaded)

## Technical Details

### Object Detection Parameters
- Confidence threshold: 0.65 for general objects, 0.45 for currency
- Frame skip: Every 2 frames are processed to reduce computational load
- Center zone: 25% around the center of the frame

### Warning System
- Alert cooldown: 6 seconds between warnings for the same object
- Announcement cooldown: 12 seconds between general object announcements
- Maximum warnings per object: 2 warnings before suppression

### Currency Recognition
- Money announcement cooldown: 5 seconds between announcements for the same denomination
- Money announcement repeat limit: 3 announcements per denomination
- Money announcement reset timeout: 15 seconds of not seeing a denomination before resetting its counter

## License

[MIT License](LICENSE)

## Contact

For questions, suggestions, or contributions, please open an issue on GitHub or contact nhaaling8910@gmail.com
