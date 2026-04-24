import heapq
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum


# ── Terrain constants (mirror mapper.py) ─────────────────────────────────────
EMPTY     = 0
FOREST    = 1
OBSTACLE  = 2
OBJECTIVE = 3
URBAN     = 4
SUPPLY    = 5

# Movement cost per terrain type (used by Dijkstra BFS)
MOVE_COST = {EMPTY: 1.0, FOREST: 2.5, OBSTACLE: 9999.0,
             OBJECTIVE: 1.0, URBAN: 1.5, SUPPLY: 1.0}

# Cover bonus [0,1]: fraction of incoming damage absorbed by terrain
COVER_BONUS = {EMPTY: 0.0, FOREST: 0.45, OBSTACLE: 0.0,
               OBJECTIVE: 0.0, URBAN: 0.30, SUPPLY: 0.10}


# ── Unit types ────────────────────────────────────────────────────────────────
class UnitType(IntEnum):
    INFANTRY = 0   # Balanced
    SNIPER   = 1   # Long range, low HP
    TANK     = 2   # High HP, short range, slow


UNIT_STATS = {
    UnitType.INFANTRY: {
        'health': 100.0, 'ammo': 30,  'fire_range': 12,
        'damage': 22.0,  'hit_chance': 0.60, 'speed': 1.0,
        'vision': 18,    'symbol': 's',      'name': 'Infantry',
    },
    UnitType.SNIPER: {
        'health': 60.0,  'ammo': 15,  'fire_range': 22,
        'damage': 40.0,  'hit_chance': 0.75, 'speed': 1.0,
        'vision': 28,    'symbol': '+',      'name': 'Sniper',
    },
    UnitType.TANK: {
        'health': 200.0, 'ammo': 20,  'fire_range': 8,
        'damage': 45.0,  'hit_chance': 0.50, 'speed': 0.6,
        'vision': 14,    'symbol': 'D',      'name': 'Tank',
    },
}


# ── Unit data class ───────────────────────────────────────────────────────────
@dataclass
class Unit:
    team:      int               # 0 = Blue (attacker), 1 = Red (defender)
    row:       int
    col:       int
    unit_type: UnitType = UnitType.INFANTRY
    health:    float = 100.0
    ammo:      int   = 30
    morale:    float = 1.0
    # Smooth animation positions (float) — visual only
    visual_row: float = 0.0
    visual_col: float = 0.0

    def __post_init__(self):
        stats = UNIT_STATS[self.unit_type]
        self.health    = stats['health']
        self.ammo      = stats['ammo']
        self.max_health = stats['health']
        self.max_ammo   = stats['ammo']
        self.visual_row = float(self.row)
        self.visual_col = float(self.col)

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def stats(self) -> dict:
        return UNIT_STATS[self.unit_type]


# ── Combat event (for visual effects) ────────────────────────────────────────
@dataclass
class CombatEvent:
    event_type: str   # 'shot', 'hit', 'kill'
    src_row: int
    src_col: int
    dst_row: int
    dst_col: int
    team: int         # attacker team


# ── Dijkstra distance map ─────────────────────────────────────────────────────
def dijkstra(terrain: np.ndarray, seeds: list) -> np.ndarray:
    """Return distance map from any seed cell, respecting terrain costs."""
    rows, cols = terrain.shape
    dist = np.full((rows, cols), np.inf)
    heap = []
    for r, c in seeds:
        if 0 <= r < rows and 0 <= c < cols:
            dist[r, c] = 0.0
            heapq.heappush(heap, (0.0, r, c))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost = MOVE_COST.get(int(terrain[nr, nc]), 1.0)
                nd = d + cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist


# ── Cellular Automaton Simulator ──────────────────────────────────────────────
class CASimulator:
    """
    Two-team tactical CA simulator with advanced behaviours.

    Tactical parameters tunable by the Rechenberg optimizer:
      aggressiveness  – probability of advancing each tick
      cover_seeking   – weight given to cover when choosing a move
      teamwork        – cohesion bonus (buffs units near allies)
    """

    AMMO_MAX    = 30
    HEALTH_MAX  = 200.0
    SPAWN_GAP   = 4    # Column spacing between spawned units

    def __init__(self, terrain: np.ndarray, params: dict = None):
        self.terrain = terrain
        self.rows, self.cols = terrain.shape

        self.params = params or {
            'aggressiveness': 0.65,
            'cover_seeking':  0.50,
            'teamwork':       0.30,
        }

        self.units: list[Unit] = []
        self.history: list[dict] = []
        self.combat_events: list[CombatEvent] = []  # Events from last step
        self.step_count = 0

        self._place_units()
        self._dist_blue = self._build_blue_dist()
        self._dist_red  = self._build_red_dist()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _place_units(self):
        # Blue team — lower-middle (52%), advances upward toward objectives
        # Composition: 60% Infantry, 25% Sniper, 15% Tank
        blue_start = int(self.rows * 0.52)
        blue_positions = []
        for r in range(blue_start, blue_start + 3):
            for c in range(0, self.cols, self.SPAWN_GAP):
                if r < self.rows and c < self.cols and self.terrain[r, c] != OBSTACLE:
                    blue_positions.append((r, c))

        for i, (r, c) in enumerate(blue_positions):
            ratio = i / max(len(blue_positions), 1)
            if ratio < 0.60:
                utype = UnitType.INFANTRY
            elif ratio < 0.85:
                utype = UnitType.SNIPER
            else:
                utype = UnitType.TANK
            self.units.append(Unit(team=0, row=r, col=c, unit_type=utype))

        # Red team — upper quarter (22%), defends the top objectives
        # Composition: 55% Infantry, 20% Sniper, 25% Tank
        red_start = int(self.rows * 0.22)
        red_positions = []
        for r in range(red_start, red_start + 3):
            for c in range(0, self.cols, self.SPAWN_GAP):
                if r < self.rows and c < self.cols and self.terrain[r, c] != OBSTACLE:
                    red_positions.append((r, c))

        for i, (r, c) in enumerate(red_positions):
            ratio = i / max(len(red_positions), 1)
            if ratio < 0.55:
                utype = UnitType.INFANTRY
            elif ratio < 0.75:
                utype = UnitType.SNIPER
            else:
                utype = UnitType.TANK
            self.units.append(Unit(team=1, row=r, col=c, unit_type=utype))

    def _build_blue_dist(self) -> np.ndarray:
        all_obj = np.argwhere(self.terrain == OBJECTIVE)
        seeds = [tuple(p) for p in all_obj if p[0] < self.rows // 2]
        if not seeds:
            seeds = [tuple(p) for p in all_obj]
        if not seeds:
            seeds = [(0, self.cols // 2)]
        return dijkstra(self.terrain, seeds)

    def _build_red_dist(self) -> np.ndarray:
        target_row = int(self.rows * 0.58)
        seeds = [(target_row, c) for c in range(0, self.cols, 2)]
        return dijkstra(self.terrain, seeds)

    # ── Simulation tick ───────────────────────────────────────────────────────

    def step(self) -> tuple:
        self.combat_events = []  # Reset events each tick
        self.step_count += 1

        alive = [u for u in self.units if u.alive]
        np.random.shuffle(alive)

        occupied = {(u.row, u.col) for u in alive}

        for u in alive:
            occupied.discard((u.row, u.col))
            self._move(u, occupied, alive)
            occupied.add((u.row, u.col))

        self._combat()
        self._supply()
        self._morale_decay()

        # Update visual positions (smooth interpolation handled by visualizer)
        for u in self.units:
            u.visual_row = float(u.row)
            u.visual_col = float(u.col)

        stats = self._collect_stats()
        self.history.append(stats)
        return self._blue_grid(), self._red_grid(), stats

    # ── Movement (BFS-guided + flanking + retreat) ────────────────────────────

    def _move(self, unit: Unit, occupied: set, all_alive: list):
        speed = unit.stats['speed']
        if np.random.random() > self.params['aggressiveness'] * speed:
            return  # Unit holds position this tick

        # Determine behaviour: retreat if low health/ammo, else advance
        should_retreat = (
            unit.health < unit.max_health * 0.25 or
            unit.ammo <= 2 or
            unit.morale < 0.15
        )

        dist_map = self._dist_blue if unit.team == 0 else self._dist_red

        # Fog of war: detect visible enemies
        vision = unit.stats['vision']
        visible_enemies = [
            e for e in all_alive
            if e.team != unit.team and e.alive
            and max(abs(e.row - unit.row), abs(e.col - unit.col)) <= vision
        ]

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        best_score = None
        best_pos   = None

        for dr, dc in dirs:
            nr, nc = unit.row + dr, unit.col + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            t = int(self.terrain[nr, nc])
            if t == OBSTACLE or (nr, nc) in occupied:
                continue
            d = dist_map[nr, nc]
            if d == np.inf:
                continue

            cover = COVER_BONUS.get(t, 0.0)

            if should_retreat:
                # Move AWAY from nearest enemy, TOWARD cover
                if visible_enemies:
                    nearest_enemy = min(
                        visible_enemies,
                        key=lambda e: abs(e.row - unit.row) + abs(e.col - unit.col)
                    )
                    enemy_dist = abs(nearest_enemy.row - nr) + abs(nearest_enemy.col - nc)
                    score = enemy_dist * 2.0 + cover * 20.0
                else:
                    score = cover * 20.0 + d * 0.5  # Just seek cover
            else:
                # Standard advance + flanking
                score = -d * self.params['aggressiveness']
                score += cover * self.params['cover_seeking'] * 12.0

                # Flanking bonus: reward positions that are to the SIDE of enemies
                if visible_enemies and unit.unit_type != UnitType.SNIPER:
                    nearest_enemy = min(
                        visible_enemies,
                        key=lambda e: abs(e.row - unit.row) + abs(e.col - unit.col)
                    )
                    # Lateral distance (perpendicular to the attack axis)
                    lateral = abs(nc - nearest_enemy.col)
                    frontal = abs(nr - nearest_enemy.row)
                    if lateral > frontal:
                        score += 3.0  # Flanking bonus

                # Cohesion: prefer cells near allies
                nearby_allies = sum(
                    1 for a in all_alive
                    if a.team == unit.team and a is not unit
                    and abs(a.row - nr) + abs(a.col - nc) <= 4
                )
                score += nearby_allies * self.params['teamwork'] * 1.5

            if best_score is None or score > best_score:
                best_score = score
                best_pos   = (nr, nc)

        if best_pos:
            unit.row, unit.col = best_pos

    # ── Combat ────────────────────────────────────────────────────────────────

    def _combat(self):
        alive_blue = [u for u in self.units if u.alive and u.team == 0]
        alive_red  = [u for u in self.units if u.alive and u.team == 1]

        for attacker in alive_blue:
            self._fire(attacker, alive_red)
        for attacker in alive_red:
            self._fire(attacker, alive_blue)

    def _fire(self, attacker: Unit, enemies: list):
        if attacker.ammo <= 0:
            attacker.morale = max(0.05, attacker.morale - 0.04)
            return

        fire_range = attacker.stats['fire_range']
        vision     = attacker.stats['vision']
        base_dmg   = attacker.stats['damage']
        hit_chance = attacker.stats['hit_chance']

        # Fog of war: can only fire at enemies within vision range
        closest, min_d = None, fire_range + 1
        for e in enemies:
            if not e.alive:
                continue
            d = max(abs(e.row - attacker.row), abs(e.col - attacker.col))
            if d <= vision and d < min_d:
                min_d, closest = d, e

        if closest is None:
            return

        attacker.ammo -= 1

        # Record shot event for visualization
        self.combat_events.append(CombatEvent(
            event_type='shot',
            src_row=attacker.row, src_col=attacker.col,
            dst_row=closest.row,  dst_col=closest.col,
            team=attacker.team,
        ))

        # Teamwork bonus: allies nearby improve accuracy
        nearby_allies = sum(
            1 for u in self.units
            if u.alive and u.team == attacker.team and u is not attacker
            and abs(u.row - attacker.row) + abs(u.col - attacker.col) <= 3
        )
        teamwork_bonus = 1.0 + self.params['teamwork'] * min(nearby_allies, 3) * 0.1

        hit_prob = (hit_chance
                    * attacker.morale
                    * (1.0 - min_d / (fire_range + 1))
                    * teamwork_bonus)

        if np.random.random() < hit_prob:
            cover  = COVER_BONUS.get(int(self.terrain[closest.row, closest.col]), 0.0)
            damage = base_dmg * (1.0 - cover) * np.random.uniform(0.7, 1.3)

            # Tanks take reduced damage
            if closest.unit_type == UnitType.TANK:
                damage *= 0.6

            closest.health -= damage
            closest.morale  = max(0.05, closest.morale - 0.08)
            attacker.morale = min(1.0,  attacker.morale + 0.02)

            if closest.health <= 0:
                self.combat_events.append(CombatEvent(
                    event_type='kill',
                    src_row=attacker.row, src_col=attacker.col,
                    dst_row=closest.row,  dst_col=closest.col,
                    team=attacker.team,
                ))
            else:
                self.combat_events.append(CombatEvent(
                    event_type='hit',
                    src_row=attacker.row, src_col=attacker.col,
                    dst_row=closest.row,  dst_col=closest.col,
                    team=attacker.team,
                ))

    # ── Supply ────────────────────────────────────────────────────────────────

    def _supply(self):
        supply_cells = set(map(tuple, np.argwhere(self.terrain == SUPPLY)))
        if not supply_cells:
            return
        for u in self.units:
            if not u.alive:
                continue
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (u.row + dr, u.col + dc) in supply_cells:
                    u.ammo   = min(u.max_ammo,    u.ammo + 6)
                    u.health = min(u.max_health,   u.health + 5.0)
                    u.morale = min(1.0,            u.morale + 0.08)
                    break

    # ── Morale decay ─────────────────────────────────────────────────────────

    def _morale_decay(self):
        for u in self.units:
            if u.alive:
                u.morale = max(0.05, u.morale - 0.008)

    # ── Visualisation helpers ─────────────────────────────────────────────────

    def _blue_grid(self) -> np.ndarray:
        g = np.zeros((self.rows, self.cols), dtype=float)
        for u in self.units:
            if u.alive and u.team == 0:
                g[u.row, u.col] = u.health / u.max_health
        return g

    def _red_grid(self) -> np.ndarray:
        g = np.zeros((self.rows, self.cols), dtype=float)
        for u in self.units:
            if u.alive and u.team == 1:
                g[u.row, u.col] = u.health / u.max_health
        return g

    # ── Statistics ────────────────────────────────────────────────────────────

    def _collect_stats(self) -> dict:
        ba = [u for u in self.units if u.alive and u.team == 0]
        ra = [u for u in self.units if u.alive and u.team == 1]
        return {
            'blue_alive':      len(ba),
            'red_alive':       len(ra),
            'blue_avg_health': float(np.mean([u.health for u in ba]) if ba else 0.0),
            'red_avg_health':  float(np.mean([u.health for u in ra]) if ra else 0.0),
            'blue_avg_ammo':   float(np.mean([u.ammo   for u in ba]) if ba else 0.0),
            'red_avg_ammo':    float(np.mean([u.ammo   for u in ra]) if ra else 0.0),
            'blue_avg_morale': float(np.mean([u.morale for u in ba]) if ba else 0.0),
            'red_avg_morale':  float(np.mean([u.morale for u in ra]) if ra else 0.0),
        }

    # ── Winner determination ──────────────────────────────────────────────────

    def get_winner(self) -> str | None:
        """
        Blue wins if any of its units reach an OBJECTIVE cell.
        Otherwise winner is determined by remaining forces.
        """
        obj_captured = any(
            u.alive and u.team == 0
            and self.terrain[u.row, u.col] == OBJECTIVE
            and u.row < self.rows // 2
            for u in self.units
        )
        if obj_captured:
            return 'Blue'

        blue_alive = sum(1 for u in self.units if u.alive and u.team == 0)
        red_alive  = sum(1 for u in self.units if u.alive and u.team == 1)

        if blue_alive == 0 and red_alive == 0:
            return 'Draw'
        if blue_alive == 0:
            return 'Red'
        if red_alive == 0:
            return 'Blue'
        return None  # Mission still in progress
