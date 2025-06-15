import numpy as np
from typing import List, Dict, Optional, Tuple
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from src.reid_bank import ReIDMemoryBank

class PlayerTracker:
    def __init__(self, min_confidence: float = 0.5, max_age: int = 30):
        """Enhanced tracker with player-specific features and re-identification"""
        self.min_confidence = min_confidence
        self.max_age = max_age

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nn_budget=50,
            max_cosine_distance=0.2,
            override_track_class=None
        )

        self.player_history = defaultdict(list)
        self.jersey_numbers = {}  # Cache for jersey numbers
        self.team_assignment = {}  # Team assignment by ID
        self.memory_bank = ReIDMemoryBank(max_memory=1000, threshold=0.3)
        
        print(f"PlayerTracker initialized (conf={min_confidence}, max_age={max_age})")

    def _validate_detection(self, detection: List[float]) -> bool:
        """Validate detection format and confidence"""
        if len(detection) < 6:
            return False
        try:
            x1, y1, x2, y2 = map(float, detection[:4])
            conf = float(detection[4])
            cls_id = int(detection[5])
            return (conf >= self.min_confidence and x1 < x2 and y1 < y2 and cls_id == 0)
        except:
            return False

    def _convert_to_deepsort_format(self, detections: List[List[float]]) -> List[Tuple[List[float], float, str]]:
        """Convert detections to DeepSort format"""
        valid_dets = []
        for det in detections:
            if not self._validate_detection(det):
                continue
            x1, y1, x2, y2, conf, _ = det
            bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
            valid_dets.append((bbox, float(conf), "player"))
        return valid_dets

    def update(self, detections: List[List[float]], frame: np.ndarray, embeddings: Optional[List[np.ndarray]] = None) -> List[List[float]]:
        """
        Update tracker with new detections and optional embeddings for re-ID.
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, conf, cls]
            frame: Current video frame
            embeddings: Optional list of embeddings for re-identification
            
        Returns:
            List of tracked players in format [x1, y1, x2, y2, track_id, conf, jersey_num, team]
        """
        if frame is None or not detections:
            return []

        valid_dets = self._convert_to_deepsort_format(detections)

        try:
            # Update tracks with new detections
            tracks = self.tracker.update_tracks(valid_dets, frame=frame)

            # Optional re-identification if embeddings are provided
            if embeddings and len(embeddings) == len(tracks):
                for i, track in enumerate(tracks):
                    if not track.is_confirmed():
                        matched_id = self.memory_bank.match(embeddings[i])
                        if matched_id is not None:
                            track.track_id = matched_id

            # Prepare results
            results = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                ltrb = track.to_ltrb()
                track_id = str(track.track_id)
                
                # Update player history
                center = ((ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2)
                self.player_history[track_id].append(center)
                if len(self.player_history[track_id]) > 30:
                    self.player_history[track_id].pop(0)

                conf = float(track.get_det_conf() or 0.0)
                conf = float(conf) if conf is not None else 0.0

                results.append([
                    int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]),
                    track_id,
                    conf,       
                    self.jersey_numbers.get(track_id, None),
                    self.team_assignment.get(track_id, None)
                ]) 
            
            return results

        except Exception as e:
            print(f"Tracking error: {str(e)[:100]}")
            return []

    def get_player_history(self, player_id: str) -> List[Tuple[float, float]]:
        """Get position history for a specific player"""
        return self.player_history.get(str(player_id), [])

    def assign_jersey_number(self, player_id: str, number: int):
        """Assign jersey number to a player"""
        self.jersey_numbers[str(player_id)] = number

    def assign_team(self, player_id: str, team: int):
        """Assign team to a player"""
        self.team_assignment[str(player_id)] = team