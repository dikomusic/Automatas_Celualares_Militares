import cv2
import numpy as np
from src.environment.grid import TacticalGrid, TerrainType

class MapParser:
    """
    Vision system for the Tactical Simulator.
    """

    def parse_image_to_grid(self, image_path: str, target_width: int = 150) -> TacticalGrid:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Intel Failure: Image at '{image_path}' not found.")

        # Resize for performance
        height, width = img.shape[:2]
        new_h = int(height * target_width / width)
        img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_NEAREST)
        
        height, width = img.shape[:2]
        grid = TacticalGrid(width=width, height=height)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # FOREST
        mask_forest = cv2.inRange(hsv, np.array([35, 40, 30]), np.array([85, 255, 255]))
        # RIVER
        mask_river  = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
        # URBAN
        mask_urban  = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 150]))

        grid.matrix[mask_urban  > 0] = TerrainType.URBAN
        grid.matrix[mask_forest > 0] = TerrainType.FOREST
        grid.matrix[mask_river  > 0] = TerrainType.RIVER 

        return grid
