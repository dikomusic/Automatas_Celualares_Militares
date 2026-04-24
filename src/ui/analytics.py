import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Any

class AnalyticsManager:
    """
    Strategic Analytics & Reporting Module.
    """

    def __init__(self):
        self.events = []
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def log_event(self, turn: int, event_type: str, faction: str, coords: tuple):
        self.events.append({'turn': turn, 'type': event_type, 'faction': faction, 'x': coords[0], 'y': coords[1]})

    def generate_mission_report(self, initial_units: List[Any], final_units: List[Any], total_turns: int):
        factions = ['blue_team', 'red_team']
        report_data = []
        print("\n" + "="*40 + "\n      TACTICAL MISSION DEBRIEF\n" + "="*40)
        for faction in factions:
            init_count = sum(1 for u in initial_units if u.faction == faction)
            end_count = sum(1 for u in final_units if u.faction == faction)
            survival_rate = (end_count / init_count * 100) if init_count > 0 else 0
            stats = {'Faction': faction, 'Survivors': end_count, 'Survival_Rate_%': round(survival_rate, 2)}
            report_data.append(stats)
            print(f"[{faction.upper()}] Survival Rate : {stats['Survival_Rate_%']}%")
        pd.DataFrame(report_data).to_csv(os.path.join(self.output_dir, "mission_report.csv"), index=False)

    def generate_heatmaps(self, original_map_path: str, grid_dims: tuple):
        base_img = cv2.imread(original_map_path)
        if base_img is None: return
        base_img = cv2.resize(base_img, grid_dims)
        h, w = grid_dims[1], grid_dims[0]

        # Killzone
        death_events = [e for e in self.events if e['type'] == 'death']
        if death_events:
            density = np.zeros((h, w), dtype=np.float32)
            for e in death_events:
                if 0 <= e['x'] < w and 0 <= e['y'] < h: density[int(e['y']), int(e['x'])] += 1
            density = cv2.GaussianBlur(density, (15, 15), 0)
            if np.max(density) > 0: density = (density / np.max(density) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(density.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.output_dir, "killzone_heatmap.png"), cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0))
