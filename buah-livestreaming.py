import os
import torch
import torchvision
from torchvision import models
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define dataset classes
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Load trained model
def load_trained_model(model_path, num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Add map_location for portability
    model.eval()
    return model

# Preprocess image
def preprocess_image(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image)

# Predict function
def predict(image, model, device):
    model.to(device)
    image = [image.to(device)]
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Draw predictions on the frame
def draw_predictions(frame, prediction, threshold=0.4):
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']
    
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.int().tolist()
            label_name = classes[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# Load model
model_path = 'fruit_detect.pth'  # <-- Replace with your model path
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = load_trained_model(model_path, num_classes)

# Start webcam and live detection
def live_stream(model, device):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting live stream... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        image_tensor = preprocess_image(frame)

        # Predict
        prediction = predict(image_tensor, model, device)

        # Draw results
        frame_with_boxes = draw_predictions(frame, prediction)

        # Display frame
        cv2.imshow('Live Fruit Detection', frame_with_boxes)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run live stream
live_stream(model, device)
