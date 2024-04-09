import streamlit as st
import cv2
import os
import torch
import numpy as np
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.dataloaders import letterbox
import yaml

st.title("YOLOv5 Object Detection with Streamlit")

def detect_objects_image(model_path, image):
    model = attempt_load(model_path)
    with open("data/coco.yaml", "r") as file:
        class_names = yaml.safe_load(file)['names']

    img = Image.open(image)
    img = np.array(img)
    img = letterbox(img, new_shape=640)[0]
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.unsqueeze(0).float() / 255.0

    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    results = []
    for det in pred[0]:
        if det is not None and len(det):
            x1, y1, x2, y2, conf, cls = det
            class_name = class_names[int(cls)]
            results.append({
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
    return results

def detect_objects_video(model_path, video_path):
    model = attempt_load(model_path)
    with open("data/coco.yaml", "r") as file:
        class_names = yaml.safe_load(file)['names']

    video_capture = cv2.VideoCapture(video_path)
    results = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = letterbox(img, new_shape=640)[0]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.unsqueeze(0).float() / 255.0

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        for det in pred[0]:
            if det is not None and len(det):
                x1, y1, x2, y2, conf, cls = det
                class_name = class_names[int(cls)]
                results.append({
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

    video_capture.release()
    return results

model_path = "yolov5s.pt"
uploaded_model = st.file_uploader("Upload YOLOv5 model", type=["pt"])
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_model:
    with open("temp_model.pt", "wb") as model_file:
        model_file.write(uploaded_model.read())
    model_path = "temp_model.pt"

if uploaded_image and uploaded_video:
    st.write("Please choose either an image or a video, not both.")

if uploaded_image:
    if st.button("Detect Objects in Image"):
        detection_results = detect_objects_image(model_path, uploaded_image)
        for result in detection_results:
            st.write(f"Detected: {result['class_name']} with confidence {result['confidence']}")

if uploaded_video:
    if st.button("Detect Objects in Video"):
        detection_results = detect_objects_video(model_path, uploaded_video)
        for result in detection_results:
            st.write(f"Detected: {result['class_name']} with confidence {result['confidence']}")

if uploaded_model:
    os.remove("temp_model.pt")
