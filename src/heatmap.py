# # src/heatmap.py
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pathlib import Path

# class HeatmapGenerator:
#     def __init__(self, output_dir="outputs/heatmaps"):
#         self.output_dir = Path(output_dir)
#         try:
#             self.output_dir.mkdir(parents=True, exist_ok=True)
#         except FileExistsError:
#             # Directory already exists, no problem
#             pass
    
#     def generate(self, positions: list, frame_num: int):
#         """positions: List of (x,y) player coordinates"""
#         plt.figure(figsize=(10, 6))
#         sns.kdeplot(x=[p[0] for p in positions], 
#                     y=[p[1] for p in positions], 
#                     cmap="Reds", shade=True)
#         output_path = self.output_dir / f"heatmap_frame_{frame_num}.png"
#         plt.savefig(output_path)
#         plt.close()
#         print(f"Heatmap saved to {output_path}")

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class HeatmapGenerator:
    def __init__(self, output_dir="outputs/heatmaps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.positions = []  # ✅ Fix: initialize positions list

    def update(self, positions):
        """Append new positions for heatmap aggregation."""
        if positions:
            self.positions.extend(positions)

    def generate(self,positions: list, frame_num: int):
        """Generate and save heatmap for all accumulated positions."""
        if not self.positions:
            print("[Heatmap] No positions to render.")
            return

        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            x=[p[0] for p in self.positions],
            y=[p[1] for p in self.positions],
            cmap="Reds",
            shade=True,
            bw_adjust=0.5
        )
        plt.axis("off")
        output_path = self.output_dir / f"heatmap_frame_{frame_num}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"[Heatmap] Saved: {output_path}")
