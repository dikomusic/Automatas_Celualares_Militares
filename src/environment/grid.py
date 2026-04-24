import numpy as np

class TerrainType:
    """
    Static constants representing movement and cover identifiers.
    """
    RIVER = -1  # Impassable barrier
    PLAIN = 1   # Open ground: Standard speed, zero cover
    URBAN = 2   # Built-up area: Moderate cost, moderate cover
    FOREST = 4  # Dense vegetation: High cost, high cover

    _COST_MAP = {
        RIVER: 999,
        PLAIN: 1,
        URBAN: 2,
        FOREST: 4
    }

    _COVER_MAP = {
        RIVER: 0.0,
        PLAIN: 0.0,
        URBAN: 0.30,
        FOREST: 0.45
    }

class TacticalGrid:
    """
    The core mathematical matrix for the Cellular Automata engine.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.matrix = np.full((height, width), TerrainType.PLAIN, dtype=int)

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def set_terrain(self, x: int, y: int, terrain_type: int) -> None:
        if self.is_in_bounds(x, y):
            self.matrix[y, x] = terrain_type

    def get_cost(self, x: int, y: int) -> int:
        if not self.is_in_bounds(x, y):
            return 999
        terrain = self.matrix[y, x]
        return TerrainType._COST_MAP.get(terrain, 1)

    def is_passable(self, x: int, y: int) -> bool:
        if not self.is_in_bounds(x, y):
            return False
        return self.matrix[y, x] != TerrainType.RIVER

    def get_cover_bonus(self, x: int, y: int) -> float:
        if not self.is_in_bounds(x, y):
            return 0.0
        terrain = self.matrix[y, x]
        return TerrainType._COVER_MAP.get(terrain, 0.0)
