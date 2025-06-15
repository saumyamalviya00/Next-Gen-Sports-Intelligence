# from ultralytics import YOLO

# # Load YOLOv8n model
# model = YOLO('yolov8n.pt')

# # Export it to ONNX
# model.export(format='onnx', opset=12, dynamic=True)

import cv2
import numpy as np
import onnxruntime as ort

class YOLOv11Detector:
    def __init__(self, model_path='models/yolov11.onnx'):
        """Initialize YOLOv11 detector with ONNX runtime"""
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            print("YOLOv11 detector initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLOv11 detector: {str(e)}")

    def detect(self, frame, conf_thresh=0.5):
        """Run detection using YOLOv11 model"""
        # Preprocess frame
        input_tensor = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs (implementation depends on your specific YOLOv11 output format)
        detections = self.process_outputs(outputs, frame.shape, conf_thresh)
        return detections

    def preprocess(self, frame):
        """Preprocess frame for YOLOv11 input"""
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = img / 255.0
        return img[np.newaxis, ...]  # Add batch dimension