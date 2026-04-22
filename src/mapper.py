import cv2
import numpy as np


class MapLoader:
    EMPTY     = 0  # Open ground
    FOREST    = 1  # Dense vegetation — cover, movement penalty
    OBSTACLE  = 2  # River / impassable terrain
    OBJECTIVE = 3  # Target/capture point (red markers)
    URBAN     = 4  # Contour/light areas — partial cover, passable
    SUPPLY    = 5  # Supply depot (yellow markers)

    def __init__(self, image_path, grid_width=150):
        self.image_path = image_path
        self.grid_width = grid_width

    def get_grid(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        height, width = img.shape[:2]
        new_h = int(height * self.grid_width / width)
        resized = cv2.resize(img, (self.grid_width, new_h), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Forest: saturated greens
        mask_forest = cv2.inRange(hsv,
                                  np.array([35,  40,  30]),
                                  np.array([85, 255, 255]))

        # River / water: cyan-blue tones
        mask_river = cv2.inRange(hsv,
                                 np.array([85,  50,  40]),
                                 np.array([135, 255, 255]))

        # Objective: red markers (wraps around 180°)
        mask_red1 = cv2.inRange(hsv, np.array([0,  120,  80]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([168, 120, 80]), np.array([180, 255, 255]))
        mask_objective = cv2.bitwise_or(mask_red1, mask_red2)

        # Urban / contour lines: low saturation, bright (the white elevation lines)
        mask_urban = cv2.inRange(hsv,
                                 np.array([0,   0, 160]),
                                 np.array([180, 45, 255]))

        # Supply: yellow markers
        mask_supply = cv2.inRange(hsv,
                                  np.array([18, 100, 100]),
                                  np.array([35, 255, 255]))

        # Build grid — order matters (later masks overwrite earlier ones)
        grid = np.zeros((new_h, self.grid_width), dtype=int)
        grid[mask_forest    > 0] = self.FOREST
        grid[mask_urban     > 0] = self.URBAN
        grid[mask_river     > 0] = self.OBSTACLE
        grid[mask_objective > 0] = self.OBJECTIVE
        grid[mask_supply    > 0] = self.SUPPLY

        return grid
