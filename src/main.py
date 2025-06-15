# Next-Gen Sports Intelligence: AI-Powered Multi-Modal Player Tracking and Real-Time Commentary

# import cv2
# import numpy as np
# import easyocr
# import time
# import json
# import pyttsx3
# from pathlib import Path
# from sklearn.cluster import KMeans
# from collections import defaultdict, deque

# # Module imports
# from src.detector import YOLOv11Detector
# from src.tracker import PlayerTracker
# from src.offside import OffsideDetector
# from src.commentary import CommentaryGenerator
# from src.heatmap import HeatmapGenerator

# def initialize_video(video_path):
#     if not Path(video_path).exists():
#         raise FileNotFoundError(f"Video not found: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Cannot open video: {video_path}")
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     return cap, width, height, fps

# def speak(text):
#     try:
#         engine = pyttsx3.init()
#         engine.say(text)
#         engine.runAndWait()
#     except:
#         pass

# def main():
#     video_path = "data/15sec_input_720p.mp4"
#     cap, width, height, fps = initialize_video(video_path)

#     detector = YOLOv11Detector('models/yolov11.onnx')
#     tracker = PlayerTracker(min_confidence=0.5, max_age=30)
#     ocr_reader = easyocr.Reader(['en'], gpu=False)
#     offside = OffsideDetector()
#     commentator = CommentaryGenerator(use_local=True)
#     heatmapper = HeatmapGenerator()
#     team_classifier = KMeans(n_clusters=2)

#     output_dir = Path("outputs")
#     output_dir.mkdir(exist_ok=True)
#     writer = cv2.VideoWriter(str(output_dir / "annotated_output.mp4"),
#                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     frame_count = 0
#     last_positions = {}
#     jersey_map = {}
#     team_map = {}
#     team_initialized = False
#     possession = {'team_1': 0, 'team_2': 0, 'neutral': 1}
#     game_events = []
#     fps_log = deque(maxlen=30)
#     commentary_cooldown = defaultdict(int)

#     print(f"Processing video: {width}x{height} @ {fps:.1f}fps")

#     while True:
#         start_time = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1

#         detections = detector.detect(frame)
#         tracks = tracker.update(detections, frame)
#         if not tracks:
#             writer.write(frame)
#             continue

#         # Jersey number recognition (updated syntax)
#         if frame_count % 10 == 0:
#             for track in tracks:
#                 bbox = track[:4]
#                 tid = track[4]
#                 x1, y1, x2, y2 = map(int, bbox)
#                 roi = frame[y1:y1+(y2-y1)//3, x1:x2]
#                 if roi.size == 0: continue
#                 gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#                 _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#                 texts = ocr_reader.readtext(th, allowlist='0123456789', min_size=10)
#                 if texts:
#                     num, conf = max(texts, key=lambda x: x[2])[:2]
#                     if conf > 0.7:
#                         jersey_map[str(tid)] = num
#                         tracker.assign_jersey_number(str(tid), num)

#         # Team initialization with fallback
#         if not team_initialized and len(tracks) >= 2:
#             colors = []
#             ids = []
#             for track in tracks:
#                 x1, y1, x2, y2, tid = map(int, track[:5])
#                 roi = frame[y1:y1+(y2-y1)//3, x1:x2]
#                 if roi.size == 0: continue
#                 colors.append(np.mean(roi, axis=(0, 1)))
#                 ids.append(tid)
            
#             if len(colors) >= 2:
#                 try:
#                     labels = team_classifier.fit_predict(colors)
#                     for tid, label in zip(ids, labels):
#                         tracker.assign_team(str(tid), label)
#                         team_map[str(tid)] = label
#                     team_initialized = True
#                     print("Teams initialized with KMeans clustering")
#                 except Exception as e:
#                     print(f"Team initialization failed: {e}")
#                     # Fallback assignment
#                     for i, track in enumerate(tracks):
#                         tid = str(track[4])
#                         team = i % 2
#                         tracker.assign_team(tid, team)
#                         team_map[tid] = team
#                     team_initialized = True
#                     print("Applied fallback team assignment")

#         # Debug output
#         if frame_count % 30 == 0:
#             print(f"\nFrame {frame_count} - Team assignments:")
#             for i, track in enumerate(tracks[:5]):  # Show first 5 players
#                 tid = str(track[4])
#                 print(f"Player {i}: ID {tid} - Team {tracker.team_assignment.get(tid, '?')}")

#         # Possession calculation
#         counts = defaultdict(int)
#         valid_players = 0
        
#         for track in tracks:
#             tid = str(track[4])
#             team = tracker.team_assignment.get(tid, team_map.get(tid, -1))
            
#             if team in (0, 1):
#                 counts[team] += 1
#                 valid_players += 1

#         total = valid_players if valid_players > 0 else 1
#         possession = {
#             'team_1': counts.get(0, 0)/total,
#             'team_2': counts.get(1, 0)/total,
#             'neutral': 1 - (counts.get(0, 0) + counts.get(1, 0))/total
#         }

#         # Player visualization
#         for track in tracks:
#             x1, y1, x2, y2 = map(int, track[:4])
#             tid = track[4]
#             jersey = track[6] if len(track) > 6 else jersey_map.get(str(tid), int(tid) % 99)
#             team = tracker.team_assignment.get(str(tid), team_map.get(str(tid), -1))
            
#             if team == 0:
#                 color = (0, 0, 255)  # Red
#                 team_text = "R"
#             elif team == 1:
#                 color = (255, 0, 0)  # Blue
#                 team_text = "B"
#             else:
#                 color = (0, 255, 0)  # Green (unassigned)
#                 team_text = "U"
            
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f"#{jersey}{team_text}", (x1, y1 - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Display FPS and possession
#         elapsed = time.time() - start_time
#         fps_log.append(1 / elapsed if elapsed > 0 else 0)
#         smooth_fps = np.mean(fps_log)
        
#         cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 25), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(frame, f"Possession R{possession['team_1']*100:.0f}% B{possession['team_2']*100:.0f}%", 
#                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Heatmap generation
#         if frame_count % int(fps) == 0:
#             positions = [last_positions[str(track[4])] for track in tracks if str(track[4]) in last_positions]
#             heatmapper.generate(positions, frame_count)

#         writer.write(frame)
#         cv2.imshow("Sports Analytics", cv2.resize(frame, (1280, 720)))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()

#     with open(output_dir / "game_log.json", 'w') as f:
#         json.dump({
#             'events': game_events,
#             'possession': possession,
#             'processing_time': frame_count/fps
#         }, f, indent=2)

#     print(f"✅ Finished: {frame_count} frames processed, logs saved.")

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import easyocr
import time
import json
import pyttsx3
from pathlib import Path
from collections import defaultdict, deque
from sklearn.cluster import KMeans

# Module imports
from src.detector import YOLOv11Detector
from src.tracker import PlayerTracker
from src.offside import OffsideDetector
from src.commentary import CommentaryGenerator
from src.heatmap import HeatmapGenerator


def initialize_video(video_path):
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cap, width, height, fps


def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[TTS Error]", e)


def main():
    video_path = "data/15sec_input_720p.mp4"
    cap, width, height, fps = initialize_video(video_path)

    detector = YOLOv11Detector("models/yolov11.onnx")
    tracker = PlayerTracker(min_confidence=0.5, max_age=30)
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    offside = OffsideDetector()
    commentator = CommentaryGenerator(use_local=True)
    heatmapper = HeatmapGenerator()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(output_dir / "annotated_output.mp4"),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    jersey_map = {}
    team_map = {}
    team_colors = {0: (0, 0, 255), 1: (255, 0, 0)}
    last_positions = {}
    position_history = defaultdict(list)
    commentary_cooldown = defaultdict(int)
    possession = {'team_1': 0, 'team_2': 0, 'neutral': 1}
    game_events = []
    fps_log = deque(maxlen=30)
    player_actions = defaultdict(lambda: {'last_pos': None, 'speed': 0})

    print(f"Processing video: {width}x{height} @ {fps:.1f}fps")

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        if not tracks:
            writer.write(frame)
            continue

        current_positions = {}
        for track in tracks:
            tid = str(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            pos = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_positions[tid] = pos
            position_history[tid].append(pos)

            if frame_count % 10 == 0 and track[5] > 0.6:
                roi = frame[y1:y1+(y2-y1)//3, x1:x2]
                if roi.size == 0:
                    continue
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                texts = ocr_reader.readtext(th, allowlist='0123456789', min_size=10)
                if texts:
                    num, conf = max(texts, key=lambda x: x[2])[:2]
                    if conf > 0.7:
                        jersey_map[tid] = num
                        tracker.assign_jersey_number(tid, num)

        last_positions.update(current_positions)

        if len(set(team_map.values())) < 2 and len(tracks) >= 4:
            team_samples = []
            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    team_samples.append((track[4], np.mean(roi, axis=(0, 1))))
            if len(team_samples) >= 4:
                colors = np.array([x[1] for x in team_samples])
                kmeans = KMeans(n_clusters=2, n_init=10).fit(colors)
                for i, (tid, _) in enumerate(team_samples):
                    team_map[str(tid)] = int(kmeans.labels_[i])

        team_counts = {0: 0, 1: 0}
        for t in tracks:
            tid = str(t[4])
            if tid in team_map:
                team_counts[team_map[tid]] += 1
        total = sum(team_counts.values())
        if total:
            possession['team_1'] = team_counts[0] / total
            possession['team_2'] = team_counts[1] / total
            possession['neutral'] = 0
        else:
            possession = {'team_1': 0, 'team_2': 0, 'neutral': 1}

        for track in tracks:
            tid = str(track[4])
            if frame_count - commentary_cooldown[tid] > 60:
                pos = current_positions[tid]
                prev = player_actions[tid]['last_pos']
                speed = 0 if not prev else np.linalg.norm(np.array(pos) - np.array(prev))
                player_actions[tid] = {'last_pos': pos, 'speed': speed}

                team = team_map.get(tid, -1)
                jersey = jersey_map.get(tid, tid)
                text = commentator.generate(
                    action_type="kick" if speed > 5 else "pass" if speed > 2 else "idle",
                    player_id=tid, jersey_num=jersey,
                    team="Red" if team == 0 else "Blue", position=pos, speed=speed)
                print("COMMENTARY:", text)
                speak(text)
                commentary_cooldown[tid] = frame_count
                game_events.append({'frame': frame_count, 'text': text})

        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = str(t[4])
            team = team_map.get(tid, -1)
            color = team_colors.get(team, (0, 255, 0))
            jersey = jersey_map.get(tid, tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'R' if team==0 else 'B'}#{jersey}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elapsed = time.time() - start
        fps_log.append(1 / elapsed if elapsed > 0 else 0)
        cv2.putText(frame, f"FPS: {np.mean(fps_log):.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Possession R{possession['team_1']*100:.0f}% B{possession['team_2']*100:.0f}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if frame_count % int(fps) == 0:
            all_positions = [p for plist in position_history.values() for p in plist]
            if all_positions:
                heatmapper.generate(all_positions, frame_count)

        writer.write(frame)
        cv2.imshow("Sports Analytics", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(output_dir / "game_log.json", 'w') as f:
        json.dump({
            'possession': possession,
            'events': game_events,
            'jersey_numbers': jersey_map,
            'team_map': team_map
        }, f, indent=2)

    print(f"✅ Finished: {frame_count} frames processed, logs saved.")


if __name__ == "__main__":
    main()

