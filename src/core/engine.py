import random
import numpy as np
from typing import List, Tuple, Dict
from src.entities.unit import Unit
from src.environment.grid import TacticalGrid
from src.core.pathfinding import Pathfinder
from src.core.combat_rules import resolve_engagement

class SimulationEngine:
    """
    The Brain of the Tactical Cellular Automata.
    """

    def __init__(
        self, 
        grid: TacticalGrid, 
        blue_team: List[Unit], 
        red_team: List[Unit], 
        supply_depots: List[Tuple[int, int]],
        objectives: Dict[str, Tuple[int, int]] = None
    ):
        self.grid = grid
        self.units = blue_team + red_team
        self.supply_depots = supply_depots
        self.pathfinder = Pathfinder()
        self.objectives = objectives or {
            'blue_team': (grid.width - 5, grid.height // 2),
            'red_team': (5, grid.height // 2)
        }

    def tick(self) -> Dict:
        """
        Executes one full iteration and returns events.
        """
        events = {'deaths': [], 'shots': []}
        random.shuffle(self.units)
        
        # Track who was alive at the start
        alive_at_start = [u for u in self.units if u.is_alive]

        for unit in alive_at_start:
            if not unit.is_alive: continue

            # 1. Rout
            if unit.morale < 0.30:
                self._process_rout(unit)
                continue

            # 2. Logistics
            if unit.ammo < 6 or unit.hp < 20:
                if self._process_logistics(unit): continue

            # 3. Suppression
            if unit.pinned_timer > 0:
                unit.pinned_timer -= 1
                self._process_combat_phase(unit, events)
                continue

            # 4. Combat
            engaged = self._process_combat_phase(unit, events)
            
            # 5. Movement
            if not engaged:
                self._process_movement(unit)

        # Post-tick cleanup and death logging
        for unit in alive_at_start:
            if not unit.is_alive:
                events['deaths'].append({'faction': unit.faction, 'x': unit.x, 'y': unit.y})
                if unit in self.units: self.units.remove(unit)

        return events

    def _process_rout(self, unit: Unit) -> None:
        nearest_enemy = self._get_nearest_enemy(unit)
        if not nearest_enemy: return
        dx = 1 if unit.x > nearest_enemy.x else -1 if unit.x < nearest_enemy.x else 0
        dy = 1 if unit.y > nearest_enemy.y else -1 if unit.y < nearest_enemy.y else 0
        tx, ty = unit.x + dx, unit.y + dy
        if self.grid.is_passable(tx, ty): unit.move_to(tx, ty)

    def _process_logistics(self, unit: Unit) -> bool:
        depot = self.pathfinder.find_nearest_supply((unit.x, unit.y), self.supply_depots, self.grid)
        if depot:
            if (unit.x, unit.y) == depot:
                unit.resupply(); return True
            path = self.pathfinder.get_optimal_path((unit.x, unit.y), depot, self.grid)
            if len(path) > 1:
                unit.move_to(path[1][0], path[1][1]); return True
        return False

    def _process_combat_phase(self, unit: Unit, events: Dict) -> bool:
        enemies = [u for u in self.units if u.faction != unit.faction and u.is_alive]
        vision = unit.vision_radius
        targets = []
        for e in enemies:
            dist = np.hypot(unit.x - e.x, unit.y - e.y)
            if dist <= vision: targets.append((dist, e))
        if targets:
            targets.sort(key=lambda x: x[0])
            target = targets[0][1]
            events['shots'].append({'attacker': (unit.x, unit.y), 'defender': (target.x, target.y), 'faction': unit.faction})
            resolve_engagement(unit, target, self.grid)
            return True
        return False

    def _process_movement(self, unit: Unit) -> None:
        if random.random() > unit.aggressiveness: return
        objective = self.objectives.get(unit.faction)
        if not objective: return
        path = self.pathfinder.get_optimal_path((unit.x, unit.y), objective, self.grid)
        if len(path) > 1:
            if random.random() < unit.cover_seeking:
                self._seek_local_cover(unit)
            else:
                unit.move_to(path[1][0], path[1][1])

    def _seek_local_cover(self, unit: Unit) -> None:
        best_pos, max_bonus = (unit.x, unit.y), self.grid.get_cover_bonus(unit.x, unit.y)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = unit.x + dx, unit.y + dy
                if self.grid.is_passable(nx, ny):
                    bonus = self.grid.get_cover_bonus(nx, ny)
                    if bonus > max_bonus: max_bonus, best_pos = bonus, (nx, ny)
        unit.move_to(best_pos[0], best_pos[1])

    def _get_nearest_enemy(self, unit: Unit) -> Unit:
        enemies = [u for u in self.units if u.faction != unit.faction]
        if not enemies: return None
        return min(enemies, key=lambda e: np.hypot(unit.x - e.x, unit.y - e.y))
