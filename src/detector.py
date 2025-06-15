# print(">>>> RUNNING CORRECT DETECTOR.PY FILE <<<<")
# import cv2
# import numpy as np
# import onnxruntime as ort

# class YOLOv11Detector:
#     def __init__(self, model_path='models/yolov11.onnx', input_size=(640, 640)):
#         # Initialize ALL attributes FIRST
#         self.input_size = input_size  # This must be first!
        
#         # Then load model
#         self.session = ort.InferenceSession(
#             model_path,
#             providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#         )
#         self.input_name = self.session.get_inputs()[0].name
#         print(f"Model initialized. Input shape: {self.input_size}")
#         print(f"Model output names: {[x.name for x in self.session.get_outputs()]}")
#         # In detector.py, add this to __init__:
#         print("Model class names:", self.session.get_outputs()[0].names)  # If available

#     def preprocess(self, frame):
#         print(f"\nPreprocessing debug:")
#         print(f"Input shape: {frame.shape} | dtype: {frame.dtype} | range: {frame.min()}-{frame.max()}")
        
#         img = cv2.resize(frame, self.input_size)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
#         img /= 255.0
#         img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
#         print(f"Resized shape: {img.shape} | normalized range: {img.min():.2f}-{img.max():.2f}")
#         return img.astype(np.float32)
    
#     def detect(self, frame, conf_thresh=0.1):
#         input_tensor = self.preprocess(frame)
#         outputs = self.session.run(None, {self.input_name: input_tensor})
    
#         # Process YOLO output (1,84,8400 format)
#         predictions = outputs[0][0]  # Shape (84,8400)
    
#         # Get all boxes where objectness > threshold
#         objectness = predictions[4:5, :]  # Objectness scores
#         mask = objectness > conf_thresh
#         filtered = predictions[:, mask.squeeze()]
    
#         # Convert to (x1,y1,x2,y2,conf,cls_id)
#         results = []
#         if filtered.size > 0:
#             # Convert from center+wh to xyxy format
#             cx = filtered[0]  # Center x
#             cy = filtered[1]  # Center y
#             w = filtered[2]   # Width
#             h = filtered[3]   # Height
#             conf = filtered[4]  # Objectness
        
#             x1 = cx - w/2
#             y1 = cy - h/2
#             x2 = cx + w/2
#             y2 = cy + h/2
        
#             # Get class ID (80-class COCO)
#             cls_id = np.argmax(filtered[5:85], axis=0)
        
#             # Scale to original image
#             h_ratio = frame.shape[0] / self.input_size[1]
#             w_ratio = frame.shape[1] / self.input_size[0]
        
#             for i in range(filtered.shape[1]):
#                 results.append([
#                     int(x1[i] * w_ratio),
#                     int(y1[i] * h_ratio),
#                     int(x2[i] * w_ratio),
#                     int(y2[i] * h_ratio),
#                     float(conf[i]),
#                     int(cls_id[i])
#                 ])
    
#         # Debug output
#         print(f"\nFiltered detections (count: {len(results)}):")
#         for i, r in enumerate(results[:5]):
#             print(f"[{i}] cls:{r[5]} conf:{r[4]:.2f} box:{r[0]},{r[1]},{r[2]},{r[3]}")
    
#         return results

import cv2
import numpy as np
import onnxruntime as ort

class YOLOv11Detector:
    def __init__(self, model_path='models/yolov11.onnx', input_size=(640, 640)):
        """Initialize YOLOv11 detector using ONNX runtime."""
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[YOLOv11Detector] Initialized | Input: {self.input_name}, Size: {self.input_size}")

    def preprocess(self, frame):
        """Resize, normalize, and reshape input frame."""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # BCHW
        return img.astype(np.float32)

    def detect(self, frame, conf_thresh=0.25):
        """Run detection and return bounding boxes."""
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0][0]  # Shape: (84, N)

        objectness = predictions[4:5, :]
        mask = objectness > conf_thresh
        filtered = predictions[:, mask.squeeze()]

        results = []
        if filtered.size > 0:
            cx, cy, w, h = filtered[0], filtered[1], filtered[2], filtered[3]
            conf = filtered[4]
            cls_id = np.argmax(filtered[5:85], axis=0)

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            h_ratio = frame.shape[0] / self.input_size[1]
            w_ratio = frame.shape[1] / self.input_size[0]

            for i in range(filtered.shape[1]):
                results.append([
                    int(x1[i] * w_ratio),
                    int(y1[i] * h_ratio),
                    int(x2[i] * w_ratio),
                    int(y2[i] * h_ratio),
                    float(conf[i]),
                    int(cls_id[i])
                ])
        return results
