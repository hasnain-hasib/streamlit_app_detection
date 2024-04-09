# YOLOv5 Object Detection with Streamlit

This Streamlit web application allows you to perform object detection using the YOLOv5 model on both images and videos. YOLOv5 is a state-of-the-art real-time object detection model that achieves high accuracy and speed.

## Usage

1. **Upload YOLOv5 Model**: Click the "Upload YOLOv5 model" button to upload your YOLOv5 model file (`.pt` format).

2. **Upload an Image**: Upload an image file (`.jpg`, `.png`, `.jpeg`) to perform object detection on a single image.

3. **Upload a Video**: Upload a video file (`.mp4`) to perform object detection on a video stream.

4. **Detect Objects in Image or Video**: Once you have uploaded either an image or a video, click the corresponding button to start the object detection process.

## Features

- **Object Detection**: Detects objects in images or videos using the YOLOv5 model.
- **Real-Time Processing**: Provides fast and efficient object detection with YOLOv5.
- **Multi-Class Detection**: Detects multiple classes of objects with their confidence levels.

## How it Works

1. **Model Loading**: The uploaded YOLOv5 model is loaded into memory.
2. **Object Detection**: The uploaded image or video is processed using the YOLOv5 model to detect objects.
3. **Display Results**: Detected objects along with their class names and confidence scores are displayed.

## Dependencies

- [Streamlit](https://streamlit.io/): Streamlit is an open-source app framework for Machine Learning and Data Science projects.
- [OpenCV](https://opencv.org/): OpenCV is a library of programming functions mainly aimed at real-time computer vision.
- [PyTorch](https://pytorch.org/): PyTorch is an open-source machine learning library used for various deep learning tasks.
- [Pillow](https://python-pillow.org/): Pillow is a Python Imaging Library (PIL) that adds image processing capabilities to your Python interpreter.

## Note

- This application assumes that the YOLOv5 model is trained and available in PyTorch's `.pt` format. You can use any YOLOv5 model variant (e.g., `yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`, `yolov5x.pt`).
- Please ensure that the uploaded image or video is compatible with the YOLOv5 model input size.


Feel free to explore, modify, and extend this application according to your requirements. If you have any questions or suggestions, please feel free to reach out!
