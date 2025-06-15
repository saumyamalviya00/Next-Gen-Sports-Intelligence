# 🏆 Next-Gen Sports Intelligence: AI-Powered Multi-Modal Player Tracking and Real-Time Commentary


**AI-powered Multi-Modal Player Re-Identification & Real-Time Commentary System**  
A real-time, intelligent sports analytics system that detects, tracks, and understands the game — with tactical overlays, team recognition, live possession stats, and AI-powered commentary.


---

## 🚀 Overview

This system is a real-time football match analyzer built using **only Python and open-source vision libraries**.  
It performs **player detection, jersey OCR, team classification, possession stats, offside detection,** and even **live commentary narration** — directly from video, fully offline.

## 🧬 System Architecture

<div align="center">

| Layer | Components | 
|-------|------------|
| 🎥 **Input** | Raw Video Feed |
| 🧠 **Detection** | YOLOv11 (Custom Sports Model) |
| 👣 **Tracking** | DeepSORT with Jersey Re-ID |
| 🔍 **Recognition** | EasyOCR → K-Means Clustering |
| 📊 **Analytics** | Possession • Offside • Actions |
| 🗣️ **Output** | Annotated Video + AI Commentary |

</div>

        
## 🔁 Data Flow
🎥 Frame Capture → OpenCV

🧠 Detection → YOLOv11 (ONNX Runtime)

👣 Tracking → DeepSORT

🔢 OCR → EasyOCR for jersey number

🎨 Clustering → KMeans for team classification

⚽ Game Intelligence → Possession & Offside logic

🗣️ Commentary → Rule-based + TTS

🎬 Overlay & Output → Annotated video & stats

## Accuracy
🎯 Player Detection: 96.4% mAP@0.5

🔢 Jersey OCR Accuracy: 88.7% (top-1)

🧢 Team Classification Accuracy: 94.2%

## 📂 Project Structure
Next-Gen Sports Intelligence/
├── src/
│   ├── detector.py       # YOLOv11 ONNX inference
│   ├── tracker.py        # DeepSORT tracking
│   ├── offside.py        # Offside detection logic
│   ├── heatmap.py        # Player movement heatmaps
│   ├── commentary.py     # TTS + rule-based commentary
│   └── main.py           # Full integrated pipeline
├── models/               # Pretrained model weights (YOLOv11)
├── data/                 # Input video files
└── outputs/              # Logs, overlays, heatmaps

## 📦 Setup and Execution
bash
Copy
Edit
pip install -r requirements.txt
python -m src.main

## 🛠️ Dependencies
Python 3.8+
OpenCV 4.5+
ONNX Runtime 1.10+
EasyOCR 1.5+
scikit-learn 1.0+
pyttsx3 (for TTS commentary)


## 📦 Deliverables
File	                 Purpose
main.py	                 Central pipeline
annotated_output.mp4	 Video with tactical overlays
game_log.json	         JSON log of possession, offside, events
heatmap_frame_*.png	     Player movement density plots

## ✨ What Makes It Unique? This isn't just another tracking script — it's a real-time AI sports analyst.
✅ Maintains player identity
✅ Tracks jersey numbers + teams
✅ Calculates live possession
✅ Detects offside positions
✅ Speaks tactical commentary it looks very distgusting improve it 
