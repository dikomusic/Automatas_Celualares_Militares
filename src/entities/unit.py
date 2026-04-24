from typing import Literal
import numpy as np

class Unit:
    """
    Base Entity class for tactical units in the Cellular Automata simulation.
    
    This class encapsulates the state, tactical traits, and physical attributes 
    of a single agent (Soldier/Vehicle) within the grid.
    """

    def __init__(
        self,
        faction: Literal['blue_team', 'red_team'],
        x: int,
        y: int,
        hp: float = 100.0,
        ammo: int = 30,
        morale: float = 1.0,
        aggressiveness: float = 0.65,
        cover_seeking: float = 0.50,
        teamwork: float = 0.30
    ):
        """
        Initialize a new tactical unit.

        Args:
            faction: The team the unit belongs to ('blue_team' or 'red_team').
            x: Horizontal coordinate (Column index in matrix).
            y: Vertical coordinate (Row index in matrix).
            hp: Hit Points (0-100).
            ammo: Remaining ammunition count.
            morale: Current morale level (0.0 to 1.0).
            aggressiveness: Propensity to advance vs hold position (0.0-1.0).
            cover_seeking: Weight given to terrain cover during movement (0.0-1.0).
            teamwork: Synergy bonus when near allies (0.0-1.0).
        """
        # Spatial Attributes
        self.faction = faction
        self.x = x  # Column
        self.y = y  # Row

        # Combat/Vital Stats
        self.hp = hp
        self.ammo = ammo
        self.morale = morale

        # Tactical Traits (Optimized via Rechenberg)
        self.aggressiveness = aggressiveness
        self.cover_seeking = cover_seeking
        self.teamwork = teamwork
        
        # State tracking
        self.pinned_timer = 0
        self.vision_radius = 12

    @property
    def is_alive(self) -> bool:
        """Check if the unit is still combat effective."""
        return self.hp > 0.0

    @property
    def position(self) -> tuple[int, int]:
        """Returns the current (x, y) coordinates."""
        return (self.x, self.y)

    def take_damage(self, amount: float) -> None:
        """
        Apply damage to the unit and reduce morale.
        
        Args:
            amount: The raw damage value to subtract from HP.
        """
        self.hp = max(0.0, self.hp - amount)
        
        # Taking fire always causes suppression/morale loss
        morale_hit = (amount / 100.0) + 0.05
        self.update_morale(-morale_hit)

    def resupply(self, ammo_gain: int = 10, hp_gain: float = 5.0) -> None:
        """
        Replenish unit resources at a supply depot.
        
        Args:
            ammo_gain: Amount of ammunition to recover.
            hp_gain: Amount of HP to recover (simulating medical aid).
        """
        self.ammo = min(30, self.ammo + ammo_gain)
        self.hp = min(100.0, self.hp + hp_gain)
        self.update_morale(0.10) # Logistics boost morale

    def update_morale(self, amount: float) -> None:
        """
        Modify the unit's morale within bounds [0.05, 1.0].
        
        Args:
            amount: Positive or negative float to adjust morale.
        """
        self.morale = np.clip(self.morale + amount, 0.05, 1.0)

    def move_to(self, new_x: int, new_y: int) -> None:
        """Update the unit's spatial coordinates."""
        self.x = new_x
        self.y = new_y

    def __repr__(self) -> str:
        return (f"Unit({self.faction}, pos=({self.x},{self.y}), "
                f"HP={self.hp:.1f}, Morale={self.morale:.2f})")
