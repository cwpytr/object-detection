import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_pred():
    def __init__(self, onnx_model, data_yaml):
        # Load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Generate class colors only once
        self.colors = self.generate_colors()

    def predictions(self, image):
        orig_h, orig_w = image.shape[:2]

        # Make image square (letterbox-style padding)
        max_size = max(orig_h, orig_w)
        input_image = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        input_image[0:orig_h, 0:orig_w] = image

        # Resize and preprocess
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Post-process
        detections = preds[0]
        boxes, confidences, class_ids = [], [], []

        x_factor = max_size / INPUT_WH_YOLO
        y_factor = max_size / INPUT_WH_YOLO

        for row in detections:
            conf = row[4]
            if conf > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[:4]

                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    confidences.append(float(conf))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
        else:
            indices = []

        for i in indices:
            x, y, w, h = boxes[i]
            label = self.labels[class_ids[i]]
            confidence = int(confidences[i] * 100)
            color = self.colors[class_ids[i]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x, y - 20), (x + w, y), color, -1)
            cv2.putText(image, f'{label}: {confidence}%', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def generate_colors(self):
        np.random.seed(42)  # For reproducible colors
        return np.random.randint(100, 255, size=(self.nc, 3)).tolist()
