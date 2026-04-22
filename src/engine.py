import heapq
import numpy as np
from dataclasses import dataclass


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


# ── Unit data class ───────────────────────────────────────────────────────────
@dataclass
class Unit:
    team:   int          # 0 = Blue (attacker), 1 = Red (defender)
    row:    int
    col:    int
    health: float = 100.0
    ammo:   int   = 30
    morale: float = 1.0

    @property
    def alive(self) -> bool:
        return self.health > 0.0


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
    Two-team tactical CA simulator.

    Tactical parameters tunable by the Rechenberg optimizer:
      aggressiveness  – probability of advancing each tick
      cover_seeking   – weight given to cover when choosing a move
      teamwork        – cohesion bonus (buffs units near allies)
    """

    FIRE_RANGE  = 12   # Chebyshev distance for fire (max of row-diff, col-diff)
    BASE_DAMAGE = 22.0
    HIT_CHANCE  = 0.60
    AMMO_MAX    = 30
    HEALTH_MAX  = 100.0
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

        self._place_units()
        self._dist_blue = self._build_blue_dist()  # Blue navigates to OBJECTIVEs
        self._dist_red  = self._build_red_dist()   # Red advances toward Blue spawn

    # ── Initialisation ────────────────────────────────────────────────────────

    def _place_units(self):
        # Blue team — lower-middle (52%), advances upward toward objectives
        blue_start = int(self.rows * 0.52)
        for r in range(blue_start, blue_start + 3):
            for c in range(0, self.cols, self.SPAWN_GAP):
                if self.terrain[r, c] != OBSTACLE:
                    self.units.append(Unit(team=0, row=r, col=c))

        # Red team — upper quarter (22%), defends the top objectives
        red_start = int(self.rows * 0.22)
        for r in range(red_start, red_start + 3):
            for c in range(0, self.cols, self.SPAWN_GAP):
                if self.terrain[r, c] != OBSTACLE:
                    self.units.append(Unit(team=1, row=r, col=c))

    def _build_blue_dist(self) -> np.ndarray:
        # Only target objectives in the top half to avoid the stray lower clusters
        all_obj = np.argwhere(self.terrain == OBJECTIVE)
        seeds = [tuple(p) for p in all_obj if p[0] < self.rows // 2]
        if not seeds:
            seeds = [tuple(p) for p in all_obj]
        if not seeds:
            seeds = [(0, self.cols // 2)]
        return dijkstra(self.terrain, seeds)

    def _build_red_dist(self) -> np.ndarray:
        # Red advances toward Blue's spawn area to intercept
        target_row = int(self.rows * 0.58)
        seeds = [(target_row, c) for c in range(0, self.cols, 2)]
        return dijkstra(self.terrain, seeds)

    # ── Simulation tick ───────────────────────────────────────────────────────

    def step(self) -> tuple:
        alive = [u for u in self.units if u.alive]
        np.random.shuffle(alive)

        occupied = {(u.row, u.col) for u in alive}

        for u in alive:
            occupied.discard((u.row, u.col))
            self._move(u, occupied)
            occupied.add((u.row, u.col))

        self._combat()
        self._supply()
        self._morale_decay()

        stats = self._collect_stats()
        self.history.append(stats)
        return self._blue_grid(), self._red_grid(), stats

    # ── Movement (BFS-guided) ─────────────────────────────────────────────────

    def _move(self, unit: Unit, occupied: set):
        if np.random.random() > self.params['aggressiveness']:
            return  # Unit holds position this tick

        dist_map = self._dist_blue if unit.team == 0 else self._dist_red
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
            # Score: reward proximity to goal AND terrain cover
            score = (-d * self.params['aggressiveness']
                     + cover * self.params['cover_seeking'] * 12.0)

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

        # Find closest enemy within fire range (Chebyshev: max of row/col diff)
        closest, min_d = None, self.FIRE_RANGE + 1
        for e in enemies:
            if not e.alive:
                continue
            d = max(abs(e.row - attacker.row), abs(e.col - attacker.col))
            if d < min_d:
                min_d, closest = d, e

        if closest is None:
            return

        attacker.ammo -= 1

        # Teamwork bonus: allies nearby improve accuracy
        nearby_allies = sum(
            1 for u in self.units
            if u.alive and u.team == attacker.team and u is not attacker
            and abs(u.row - attacker.row) + abs(u.col - attacker.col) <= 3
        )
        teamwork_bonus = 1.0 + self.params['teamwork'] * min(nearby_allies, 3) * 0.1

        hit_prob = (self.HIT_CHANCE
                    * attacker.morale
                    * (1.0 - min_d / self.FIRE_RANGE)
                    * teamwork_bonus)

        if np.random.random() < hit_prob:
            cover  = COVER_BONUS.get(int(self.terrain[closest.row, closest.col]), 0.0)
            damage = self.BASE_DAMAGE * (1.0 - cover) * np.random.uniform(0.7, 1.3)
            closest.health -= damage
            closest.morale  = max(0.05, closest.morale - 0.08)
            attacker.morale = min(1.0,  attacker.morale + 0.02)

    # ── Supply ────────────────────────────────────────────────────────────────

    def _supply(self):
        supply_cells = set(map(tuple, np.argwhere(self.terrain == SUPPLY)))
        if not supply_cells:
            return
        for u in self.units:
            if not u.alive:
                continue
            # Check unit's cell and immediate neighbours
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (u.row + dr, u.col + dc) in supply_cells:
                    u.ammo   = min(self.AMMO_MAX,   u.ammo + 6)
                    u.health = min(self.HEALTH_MAX, u.health + 5.0)
                    u.morale = min(1.0,             u.morale + 0.08)
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
                g[u.row, u.col] = u.health / self.HEALTH_MAX
        return g

    def _red_grid(self) -> np.ndarray:
        g = np.zeros((self.rows, self.cols), dtype=float)
        for u in self.units:
            if u.alive and u.team == 1:
                g[u.row, u.col] = u.health / self.HEALTH_MAX
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
        # Only count top-half objectives as valid capture targets
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
