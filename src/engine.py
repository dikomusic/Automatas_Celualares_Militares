import numpy as np
from scipy.signal import convolve2d

class CASimulator:
    def __init__(self, terrain_grid):
        self.terrain = terrain_grid
        # Unit Layer: 0 = No Unit, 1 = Team Blue (Attacking)
        self.units = np.zeros_like(terrain_grid, dtype=float)
        
        # Initialize some units at the bottom of the map
        self.units[-5:, :] = 1.0 

    def step(self):
        """One tick of the simulation using Vectorized Rules"""
        # 1. Count neighbors using a 3x3 kernel (Moore Neighborhood)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve2d(self.units, kernel, mode='same', boundary='fill')

        # 2. Define Transition Rules
        # Rule A: Units spread to empty adjacent cells (Diffusion)
        # Rule B: Movement is blocked by CONTOUR (State 2)
        # Rule C: Movement is slowed by FOREST (State 1)
        
        next_units = np.copy(self.units)
        
        # Stochastic spread: Probability of moving into a cell increases with neighbors
        # and decreases based on terrain cost.
        move_chance = np.random.random(self.units.shape)
        
        # Logic: If empty AND has neighbors AND (Chance > Terrain Penalty)
        can_move = (self.units == 0) & (neighbor_count > 0)
        
        # Terrain Penalties: Contours (2) are hard walls, Forests (1) are 50% slower
        terrain_penalty = np.ones_like(self.units)
        terrain_penalty[self.terrain == 2] = 1.1 # Impossible to cross
        terrain_penalty[self.terrain == 1] = 0.6 # Forest penalty
        
        # Apply the update
        success = (move_chance < (neighbor_count * 0.1 * terrain_penalty))
        next_units[can_move & success] = 1.0
        
        self.units = next_units
        return self.units
