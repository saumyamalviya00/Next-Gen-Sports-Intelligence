# import numpy as np
# from typing import Dict, List

# class OffsideDetector:
#     def __init__(self):
#         """
#         Robust offside detection with team assignment and cooldown
#         """
#         self.team_assignment = {}  # {player_id: "attack"/"defense"}
#         self.last_offside_frame = -100
#         self.current_frame = 0
#         self.min_players_for_teams = 4  # Minimum players to attempt team assignment

#     def _validate_tracks(self, tracks: Dict[int, list]) -> Dict[int, list]:
#         """Validate and filter track data"""
#         valid_tracks = {}
#         for track_id, bbox in tracks.items():
#             try:
#                 if len(bbox) != 4:
#                     continue
                    
#                 x1, y1, x2, y2 = map(float, bbox)
#                 if any(np.isnan(v) or any(np.isinf(v)) for v in [x1, y1, x2, y2]):
#                     continue
                    
#                 valid_tracks[track_id] = [x1, y1, x2, y2]
#             except Exception as e:
#                 print(f"Invalid track {track_id}: {e}")
#         return valid_tracks

#     def assign_teams(self, tracks: Dict[int, list]):
#         """
#         Robust team assignment based on field position
#         Args:
#             tracks: {player_id: [x1,y1,x2,y2]}
#         """
#         if len(tracks) < self.min_players_for_teams:
#             return
            
#         try:
#             # Sort players by x-position (left to right = defense to attack)
#             sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x][0])
            
#             # Alternate team assignment with buffer zone
#             mid_point = len(sorted_ids) // 2
#             for i, player_id in enumerate(sorted_ids):
#                 self.team_assignment[player_id] = "attack" if i > mid_point else "defense"
#         except Exception as e:
#             print(f"Team assignment failed: {e}")

#     def check(self, tracks: Dict[int, list]) -> List[int]:
#         """
#         Detect offside players with validation and cooldown
#         Args:
#             tracks: {player_id: [x1,y1,x2,y2]}
#         Returns:
#             List of offside player IDs
#         """
#         self.current_frame += 1
#         valid_tracks = self._validate_tracks(tracks)
#         if not valid_tracks:
#             return []

#         # Cooldown check
#         if self.current_frame - self.last_offside_frame < 5:
#             return []

#         # Team assignment
#         self.assign_teams(valid_tracks)
#         if not self.team_assignment:
#             return []

#         # Get attackers and defenders
#         try:
#             attackers = [id for id in valid_tracks 
#                         if self.team_assignment.get(id) == "attack"]
#             defenders = [id for id in valid_tracks 
#                         if self.team_assignment.get(id) == "defense"]
            
#             if not defenders or not attackers:
#                 return []

#             # Find last defender (minimum x-coordinate)
#             last_defender = min(valid_tracks[d][0] for d in defenders)
            
#             # Check attackers beyond last defender
#             offsides = [
#                 a for a in attackers 
#                 if valid_tracks[a][0] < last_defender  # x1 < last defender's x1
#             ]
            
#             if offsides:
#                 self.last_offside_frame = self.current_frame
#                 print(f"Offside detected - players: {offsides}")
#                 return offsides
                
#         except Exception as e:
#             print(f"Offside check failed: {e}")
            
#         return []

import numpy as np
from typing import Dict, List

class OffsideDetector:
    def __init__(self):
        self.team_assignment = {}        # player_id: 'attack' or 'defense'
        self.last_offside_frame = -100
        self.current_frame = 0
        self.cooldown = 5
        self.min_players_for_teams = 4

    def _validate_tracks(self, tracks: Dict[int, list]) -> Dict[int, list]:
        valid_tracks = {}
        for tid, bbox in tracks.items():
            if len(bbox) == 4 and all(np.isfinite(bbox)):
                valid_tracks[tid] = bbox
        return valid_tracks

    def assign_teams(self, tracks: Dict[int, list]):
        """Assign players to teams based on field x-position."""
        if len(tracks) < self.min_players_for_teams:
            return

        sorted_ids = sorted(tracks, key=lambda pid: tracks[pid][0])
        mid = len(sorted_ids) // 2
        for i, pid in enumerate(sorted_ids):
            self.team_assignment[pid] = "attack" if i > mid else "defense"

    def check(self, tracks: Dict[int, list]) -> List[int]:
        self.current_frame += 1
        if self.current_frame - self.last_offside_frame < self.cooldown:
            return []

        valid_tracks = self._validate_tracks(tracks)
        if not valid_tracks:
            return []

        self.assign_teams(valid_tracks)

        attackers = [pid for pid in valid_tracks if self.team_assignment.get(pid) == "attack"]
        defenders = [pid for pid in valid_tracks if self.team_assignment.get(pid) == "defense"]

        if not attackers or not defenders:
            return []

        last_defender_x = min(valid_tracks[d][0] for d in defenders)
        offsides = [a for a in attackers if valid_tracks[a][0] < last_defender_x]

        if offsides:
            self.last_offside_frame = self.current_frame
            print(f"[Offside] Players: {offsides}")

        return offsides
