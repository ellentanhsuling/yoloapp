# YOLOapp - Mobile Object Detection

YOLOapp is a Streamlit-based web application that leverages the YOLO (You Only Look Once) object detection model to identify objects in real-time using your iPhone's built-in camera.

![YOLOapp Demo](https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg)

## Features

- Real-time object detection using YOLOv8
- Mobile-friendly interface optimized for iPhone use
- Camera access through the web browser
- Adjustable confidence threshold for detections
- Display of bounding boxes and labels on detected objects
- Detection result summaries with confidence scores
- Performance metrics like inference time

## Requirements

- Python 3.8+
- Streamlit 1.22.0+
- Ultralytics YOLOv8
- OpenCV
- Numpy
- Pillow

## Installation

1. Clone this repository:ß
```bash
git clone https://github.com/yourusername/YOLOapp.git
cd YOLOapp
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Local Development
When running locally, access the app at http://localhost:8501 on your computer.

### Accessing from iPhone
To use the app on your iPhone:

1. Run the app on a machine in your local network
2. Find your computer's IP address
3. On your iPhone, open a browser and navigate to http://[YOUR_IP_ADDRESS]:8501
4. Allow camera permissions when prompted
5. Use the interface to take pictures and detect objects

### Deployment Options
For permanent access, deploy the app to:
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [AWS](https://aws.amazon.com)
- Any other platform that supports Python web applications

## How It Works

1. The app uses Streamlit's camera_input function to access your device's camera
2. When you take a picture, it's processed by the YOLOv8 model
3. The model identifies objects in the image and returns bounding boxes and class predictions
4. The app displays the results with visual indicators and text descriptions

## Troubleshooting

### Camera Not Working on iPhone
- Ensure your iPhone and computer are on the same network
- Make sure you've allowed camera permissions in your browser
- For security reasons, camera access typically requires HTTPS or localhost
- Some browsers may have restrictions on camera access

### Slow Performance
- The default model (YOLOv8n) is optimized for speed, but performance depends on your server
- For faster processing, consider:
  - Using a smaller model version
  - Running on a machine with better hardware
  - Reducing the image resolution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [Streamlit](https://streamlit.io/) for the web application framework

ß