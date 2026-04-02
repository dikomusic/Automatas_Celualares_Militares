import cv2
import numpy as np

class MapLoader:
    def __init__(self, image_path, grid_width=200):
        self.image_path = image_path
        self.grid_width = grid_width

        # State Constants
        self.EMPTY = 0
        self.FOREST = 1
        self.CONTOUR = 2
        self.OBJECTIVE = 3

    def get_grid(self):
        img = cv2.imread(self.image_path)
        if img is None: raise FileNotFoundError(f"Could not find {self.image_path}")

        height, width = img.shape[:2]
        ratio = self.grid_width / width
        grid_size = (self.grid_width, int(height * ratio))
        resized = cv2.resize(img, grid_size, interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Color Segmentation
        mask_green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_contours = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))

        grid = np.zeros(grid_size[::-1], dtype=int)
        grid[mask_green > 0] = self.FOREST
        grid[mask_contours > 0] = self.CONTOUR
        grid[mask_red > 0] = self.OBJECTIVE
        return grid
