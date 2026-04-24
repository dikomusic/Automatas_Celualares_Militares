import heapq
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

# ── Terrain constants ─────────────────────────────────────────────────────────
EMPTY     = 0
FOREST    = 1
OBSTACLE  = 2
OBJECTIVE = 3
URBAN     = 4
SUPPLY    = 5

MOVE_COST = {
    EMPTY: 1.0, FOREST: 2.5, OBSTACLE: 9999.0,
    OBJECTIVE: 1.0, URBAN: 1.8, SUPPLY: 1.0,
}
COVER_BONUS = {
    EMPTY: 0.0, FOREST: 0.45, OBSTACLE: 0.0,
    OBJECTIVE: 0.05, URBAN: 0.30, SUPPLY: 0.10,
}
# Approximate elevation (Rule 2 – slope fatigue)
ELEVATION = {EMPTY: 0, FOREST: 1, OBSTACLE: 0, OBJECTIVE: 0, URBAN: 2, SUPPLY: 0}

TERRAIN_NAMES = {
    EMPTY: 'Vacio', FOREST: 'Bosque', OBSTACLE: 'Obstaculo',
    OBJECTIVE: 'Objetivo', URBAN: 'Urbano', SUPPLY: 'Suministro',
}

# ── Weather (Rule 18 / environmental modifier) ────────────────────────────────
class Weather(IntEnum):
    CLEAR = 0
    RAIN  = 1
    FOG   = 2
    COLD  = 3
    HEAT  = 4

WEATHER_NAMES = {
    Weather.CLEAR: 'Despejado',
    Weather.RAIN:  'Lluvia',
    Weather.FOG:   'Niebla',
    Weather.COLD:  'Frio',
    Weather.HEAT:  'Calor',
}

# Per-weather modifiers applied each step
WEATHER_FX = {
    Weather.CLEAR: dict(vision_mult=1.00, move_mult=1.00, morale_decay=0.008, hit_penalty=0.00),
    Weather.RAIN:  dict(vision_mult=0.55, move_mult=0.80, morale_decay=0.013, hit_penalty=0.12),
    Weather.FOG:   dict(vision_mult=0.30, move_mult=0.90, morale_decay=0.010, hit_penalty=0.22),
    Weather.COLD:  dict(vision_mult=0.90, move_mult=0.65, morale_decay=0.016, hit_penalty=0.06),
    Weather.HEAT:  dict(vision_mult=0.80, move_mult=0.85, morale_decay=0.020, hit_penalty=0.06),
}

# ── 6 Arms of the Bolivian Army ───────────────────────────────────────────────
class UnitType(IntEnum):
    INFANTRY       = 0   # Infantería   – balanced attacker
    ARTILLERY      = 1   # Artillería   – long range, area suppression
    CAVALRY        = 2   # Caballería   – double move speed
    COMMUNICATIONS = 3   # Comunicaciones – comms link bonus to allies
    ENGINEERING    = 4   # Ingeniería   – clears / creates obstacles
    LOGISTICS      = 5   # Logística    – mobile resupply

UNIT_STATS = {
    UnitType.INFANTRY: dict(
        health=100, ammo=30, fire_range=12, damage=22, hit_chance=0.60,
        speed=1.0, vision=18, area=0, comms_range=0, supply_range=0,
        name='Infanteria',
    ),
    UnitType.ARTILLERY: dict(
        health=80,  ammo=15, fire_range=28, damage=55, hit_chance=0.55,
        speed=0.5,  vision=14, area=2, comms_range=0, supply_range=0,
        name='Artilleria',
    ),
    UnitType.CAVALRY: dict(
        health=110, ammo=20, fire_range=10, damage=28, hit_chance=0.65,
        speed=2.0,  vision=20, area=0, comms_range=0, supply_range=0,
        name='Caballeria',
    ),
    UnitType.COMMUNICATIONS: dict(
        health=70,  ammo=10, fire_range=8,  damage=10, hit_chance=0.40,
        speed=0.8,  vision=24, area=0, comms_range=18, supply_range=0,
        name='Comunicaciones',
    ),
    UnitType.ENGINEERING: dict(
        health=90,  ammo=20, fire_range=8,  damage=18, hit_chance=0.50,
        speed=0.7,  vision=16, area=0, comms_range=0, supply_range=0,
        name='Ingenieria',
    ),
    UnitType.LOGISTICS: dict(
        health=75,  ammo=5,  fire_range=6,  damage=8,  hit_chance=0.35,
        speed=0.6,  vision=14, area=0, comms_range=0, supply_range=12,
        name='Logistica',
    ),
}

# ── Unit states ───────────────────────────────────────────────────────────────
class UnitState(IntEnum):
    ACTIVE       = 0   # Normal operation
    DEFENSIVE    = 1   # Blocking – immobile, −50% damage received (Rule 6)
    SUPPRESSED   = 2   # Artillery hit – can't move/attack (Rule 18)
    FIXED        = 3   # Pinned by fire – can't move (Rule 8)
    INHIBITED    = 4   # Constant fire – can't attack (Rule 11)
    RETREATING   = 5   # HP < 30% – moves toward own lines (Rule 19)
    REORGANIZING = 6   # Post-capture 2-tick pause (Rule 20)
    INACTIVE     = 7   # Strategic reserve – activates on threshold (Rule 24)

STATE_NAMES = {
    UnitState.ACTIVE:       'Activo',
    UnitState.DEFENSIVE:    'Defensivo',
    UnitState.SUPPRESSED:   'Suprimido',
    UnitState.FIXED:        'Fijado',
    UnitState.INHIBITED:    'Inhibido',
    UnitState.RETREATING:   'Retirada',
    UnitState.REORGANIZING: 'Reorganiz.',
    UnitState.INACTIVE:     'Reserva',
}

# ── Unit ──────────────────────────────────────────────────────────────────────
@dataclass
class Unit:
    team:      int
    row:       int
    col:       int
    unit_type: UnitType  = UnitType.INFANTRY
    morale:    float     = 1.0
    state:     UnitState = UnitState.ACTIVE
    # State timers (ticks remaining)
    suppress_timer: int = 0
    fixed_timer:    int = 0
    reorg_timer:    int = 0
    build_timer:    int = 0       # Engineering: ticks spent working on a cell
    departure_step: int = 0       # Rule 5: can't advance before this step
    hits_taken:     int = 0       # Rule 11: neutralization counter
    # Set by __post_init__
    health:     float = 0.0
    ammo:       int   = 0
    max_health: float = 0.0
    max_ammo:   int   = 0
    visual_row: float = 0.0
    visual_col: float = 0.0

    def __post_init__(self):
        s = UNIT_STATS[self.unit_type]
        if self.health == 0.0:
            self.health = float(s['health'])
        if self.ammo == 0:
            self.ammo = int(s['ammo'])
        self.max_health = float(s['health'])
        self.max_ammo   = int(s['ammo'])
        self.visual_row = float(self.row)
        self.visual_col = float(self.col)

    @property
    def alive(self) -> bool:
        return self.health > 0.0 and self.state != UnitState.INACTIVE

    @property
    def stats(self) -> dict:
        return UNIT_STATS[self.unit_type]


# ── Combat event (for visual FX) ─────────────────────────────────────────────
@dataclass
class CombatEvent:
    event_type: str    # 'shot', 'hit', 'kill', 'suppress', 'build'
    src_row: int
    src_col: int
    dst_row: int
    dst_col: int
    team: int


# ── Dijkstra weighted BFS ─────────────────────────────────────────────────────
def dijkstra(terrain: np.ndarray, seeds: list) -> np.ndarray:
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
                nd = d + MOVE_COST.get(int(terrain[nr, nc]), 1.0)
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return dist


# ── Cellular Automaton Simulator ──────────────────────────────────────────────
class CASimulator:
    """
    25-rule military CA simulator — 6 arms of the Bolivian Army.

    Tactical params (tunable by Rechenberg):
      aggressiveness, cover_seeking, teamwork, departure_step
    """

    SPAWN_GAP            = 4
    COMMS_HIT_PENALTY    = 0.20   # Rule 16: out-of-comms accuracy penalty
    ENG_BUILD_TICKS      = 3      # Rule 14/15: ticks to clear or build obstacle
    ISOLATION_RADIUS     = 3      # Rule 21: radius to check surrounding enemies
    SUPPRESSION_TICKS    = 2      # Rule 18
    FIXATION_TICKS       = 1      # Rule 8
    INHIBIT_TICKS        = 2      # Rule 11
    REORG_TICKS          = 2      # Rule 20
    NEUTRALIZE_THRESHOLD = 3      # Rule 11: hits before inhibited
    RESERVE_THRESHOLD    = 0.60   # Rule 24: activate if alive < 60% of initial

    def __init__(self, terrain: np.ndarray, params: dict = None,
                 weather: Weather = Weather.CLEAR,
                 custom_units: list = None):
        self.terrain      = terrain.copy().astype(int)  # mutable (engineering)
        self.base_terrain = terrain.copy().astype(int)  # original
        self.rows, self.cols = terrain.shape
        self.weather      = weather
        self.params       = params or {
            'aggressiveness': 0.65,
            'cover_seeking':  0.50,
            'teamwork':       0.30,
            'departure_step': 0,
        }
        self.units: list[Unit]              = []
        self.combat_events: list[CombatEvent] = []
        self.history: list[dict]            = []
        self.step_count = 0

        # Track initial counts for reserve activation
        self._blue_initial = 0
        self._red_initial  = 0

        if custom_units:
            self.units = list(custom_units)
        else:
            self._place_units()

        self._blue_initial = sum(1 for u in self.units if u.team == 0 and u.state != UnitState.INACTIVE)
        self._red_initial  = sum(1 for u in self.units if u.team == 1 and u.state != UnitState.INACTIVE)

        self._rebuild_dist()

    # ── Placement ─────────────────────────────────────────────────────────────

    def _place_units(self):
        dep = int(self.params.get('departure_step', 0))

        # Composition: 45% Inf, 12% Art, 18% Cav, 8% Com, 10% Eng, 7% Log
        arm_thresholds = [
            (0.45, UnitType.INFANTRY),
            (0.57, UnitType.ARTILLERY),
            (0.75, UnitType.CAVALRY),
            (0.83, UnitType.COMMUNICATIONS),
            (0.93, UnitType.ENGINEERING),
            (1.00, UnitType.LOGISTICS),
        ]

        def spawn_row(frac, team):
            start = int(self.rows * frac)
            positions = []
            for r in range(start, min(start + 3, self.rows)):
                for c in range(0, self.cols, self.SPAWN_GAP):
                    if self.terrain[r, c] != OBSTACLE:
                        positions.append((r, c))
            for i, (r, c) in enumerate(positions):
                ratio = i / max(len(positions), 1)
                utype = UnitType.INFANTRY
                for thr, ut in arm_thresholds:
                    if ratio < thr:
                        utype = ut
                        break
                u = Unit(team=team, row=r, col=c, unit_type=utype)
                if team == 0:
                    u.departure_step = dep
                self.units.append(u)

        spawn_row(0.52, 0)   # Blue: lower half
        spawn_row(0.22, 1)   # Red:  upper area

    # ── Distance maps ─────────────────────────────────────────────────────────

    def _rebuild_dist(self):
        self._dist_blue = self._build_blue_dist()
        self._dist_red  = self._build_red_dist()

    def _build_blue_dist(self):
        all_obj = np.argwhere(self.terrain == OBJECTIVE)
        seeds = [(r, c) for r, c in all_obj if r < self.rows // 2]
        if not seeds:
            seeds = [(r, c) for r, c in all_obj]
        if not seeds:
            seeds = [(0, self.cols // 2)]
        return dijkstra(self.terrain, seeds)

    def _build_red_dist(self):
        target = int(self.rows * 0.58)
        seeds = [(target, c) for c in range(0, self.cols, 2)]
        return dijkstra(self.terrain, seeds)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self) -> tuple:
        self.combat_events = []
        self.step_count += 1
        wx = WEATHER_FX[self.weather]

        alive = [u for u in self.units if u.alive]
        np.random.shuffle(alive)
        occupied = {(u.row, u.col) for u in alive}

        # ── Decrement state timers ────────────────────────────────────────
        for u in alive:
            if u.suppress_timer > 0:
                u.suppress_timer -= 1
                if u.suppress_timer == 0 and u.state == UnitState.SUPPRESSED:
                    u.state = UnitState.ACTIVE
            if u.fixed_timer > 0:
                u.fixed_timer -= 1
                if u.fixed_timer == 0 and u.state == UnitState.FIXED:
                    u.state = UnitState.ACTIVE
            if u.reorg_timer > 0:
                u.reorg_timer -= 1
                if u.reorg_timer == 0 and u.state == UnitState.REORGANIZING:
                    u.state = UnitState.ACTIVE

        # Rule 24: activate reserves if front drops below threshold
        self._activate_reserves(alive)

        # ── Movement ──────────────────────────────────────────────────────
        for u in alive:
            occupied.discard((u.row, u.col))
            self._move(u, occupied, alive, wx)
            occupied.add((u.row, u.col))

        # Rule 14/15: Engineering builds/clears
        self._engineering_actions(alive)

        # ── Combat ────────────────────────────────────────────────────────
        blue_a = [u for u in alive if u.team == 0]
        red_a  = [u for u in alive if u.team == 1]
        for att in blue_a:
            self._fire(att, red_a, alive, wx)
        for att in red_a:
            self._fire(att, blue_a, alive, wx)

        # ── Logistics resupply (Rule 17) ──────────────────────────────────
        self._logistics_resupply(alive)

        # ── Supply depot ──────────────────────────────────────────────────
        self._supply_depot(alive)

        # ── Logistic isolation decay (Rule 21) ────────────────────────────
        self._isolation_decay(alive)

        # ── Morale & weather decay ─────────────────────────────────────────
        for u in alive:
            u.morale = max(0.05, u.morale - wx['morale_decay'])

        # Rule 20: post-capture reorganization trigger
        self._check_reorganize()

        # Smooth visuals
        for u in self.units:
            u.visual_row = float(u.row)
            u.visual_col = float(u.col)

        stats = self._collect_stats()
        self.history.append(stats)
        return self._blue_grid(), self._red_grid(), stats

    # ── Rule 24: strategic reserve activation ─────────────────────────────────

    def _activate_reserves(self, alive: list):
        blue_alive = sum(1 for u in alive if u.team == 0)
        red_alive  = sum(1 for u in alive if u.team == 1)
        for u in self.units:
            if u.state != UnitState.INACTIVE:
                continue
            ref = self._blue_initial if u.team == 0 else self._red_initial
            cur = blue_alive if u.team == 0 else red_alive
            if ref > 0 and cur < ref * self.RESERVE_THRESHOLD:
                u.state = UnitState.ACTIVE

    # ── Rule 1–5, 13, 17, 19: movement ───────────────────────────────────────

    def _move(self, unit: Unit, occupied: set, all_alive: list, wx: dict):
        # States that prevent movement
        if unit.state in (UnitState.SUPPRESSED, UnitState.REORGANIZING,
                          UnitState.DEFENSIVE, UnitState.FIXED):
            return

        # Rule 5: line of departure
        if self.step_count < unit.departure_step:
            return

        # Rule 13: cavalry moves twice per tick; others once
        move_steps = 2 if unit.unit_type == UnitType.CAVALRY else 1

        speed = unit.stats['speed'] * wx['move_mult']
        if np.random.random() > self.params['aggressiveness'] * speed:
            return

        # Rule 19: retreat state management
        if unit.health < unit.max_health * 0.30 or unit.morale < 0.12:
            unit.state = UnitState.RETREATING
        elif unit.state == UnitState.RETREATING and unit.health > unit.max_health * 0.55:
            unit.state = UnitState.ACTIVE

        dist_map = self._dist_blue if unit.team == 0 else self._dist_red

        # Rule 3/22: vision modified by weather
        vision = max(3, int(unit.stats['vision'] * wx['vision_mult']))
        # Rule 4: elevated observation bonus from URBAN terrain
        if self.terrain[unit.row, unit.col] == URBAN:
            vision = int(vision * 1.25)

        visible_enemies = [
            e for e in all_alive
            if e.team != unit.team and e.alive
            and max(abs(e.row - unit.row), abs(e.col - unit.col)) <= vision
        ]

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1,-1), (-1, 1), (1,-1), (1, 1)]

        for _ in range(move_steps):
            best_score, best_pos = None, None

            for dr, dc in dirs:
                nr, nc = unit.row + dr, unit.col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                t = int(self.terrain[nr, nc])
                if t == OBSTACLE or (nr, nc) in occupied:
                    continue
                # Rule 6: defensive enemy physically blocks the cell
                if any(e.row == nr and e.col == nc
                       and e.state == UnitState.DEFENSIVE
                       for e in all_alive if e.team != unit.team):
                    continue
                d = dist_map[nr, nc]
                if d == np.inf:
                    continue

                cover = COVER_BONUS.get(t, 0.0)
                # Rule 2: slope fatigue
                elev_diff = ELEVATION.get(t, 0) - ELEVATION.get(
                    int(self.terrain[unit.row, unit.col]), 0)
                slope_pen = max(0, elev_diff) * 0.3

                if unit.state == UnitState.RETREATING:
                    if visible_enemies:
                        nearest = min(visible_enemies,
                                      key=lambda e: abs(e.row-unit.row)+abs(e.col-unit.col))
                        score = (abs(nearest.row - nr) + abs(nearest.col - nc)) * 2.5
                        score += cover * 22.0
                    else:
                        score = cover * 20.0 + d * 0.5
                else:
                    score = -d * self.params['aggressiveness'] - slope_pen
                    score += cover * self.params['cover_seeking'] * 12.0

                    # Flanking bonus (Rule 7: channeling implicitly)
                    if visible_enemies and unit.unit_type not in (
                            UnitType.ARTILLERY, UnitType.LOGISTICS):
                        nearest = min(visible_enemies,
                                      key=lambda e: abs(e.row-unit.row)+abs(e.col-unit.col))
                        if abs(nc - nearest.col) > abs(nr - nearest.row):
                            score += 3.0

                    # Cohesion
                    nearby = sum(
                        1 for a in all_alive
                        if a.team == unit.team and a is not unit
                        and abs(a.row - nr) + abs(a.col - nc) <= 4
                    )
                    score += nearby * self.params['teamwork'] * 1.5

                    # Rule 17: low ammo → move toward logistics
                    if unit.ammo < unit.max_ammo * 0.20:
                        logs = [x for x in all_alive
                                if x.team == unit.team
                                and x.unit_type == UnitType.LOGISTICS]
                        if logs:
                            nd = min(abs(x.row-nr)+abs(x.col-nc) for x in logs)
                            score += max(0, 20 - nd) * 1.5

                if best_score is None or score > best_score:
                    best_score = score
                    best_pos = (nr, nc)

            if best_pos:
                occupied.discard((unit.row, unit.col))
                unit.row, unit.col = best_pos
                occupied.add((unit.row, unit.col))
            else:
                break   # Cavalry: stop if no valid move

    # ── Rule 16: communications link ─────────────────────────────────────────

    def _has_comms_link(self, unit: Unit, all_alive: list) -> bool:
        comms = [u for u in all_alive
                 if u.team == unit.team
                 and u.unit_type == UnitType.COMMUNICATIONS
                 and u is not unit and u.alive]
        if not comms:
            return True  # No comms unit on team → no penalty applied
        for cu in comms:
            if max(abs(cu.row - unit.row), abs(cu.col - unit.col)) <= cu.stats['comms_range']:
                return True
        return False

    # ── Rules 14 & 15: engineering actions ───────────────────────────────────

    def _engineering_actions(self, alive: list):
        for u in alive:
            if u.unit_type != UnitType.ENGINEERING:
                continue
            worked = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = u.row + dr, u.col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Rule 14: clear obstacle
                if self.terrain[nr, nc] == OBSTACLE:
                    u.build_timer += 1
                    if u.build_timer >= self.ENG_BUILD_TICKS:
                        self.terrain[nr, nc] = EMPTY
                        u.build_timer = 0
                        self._rebuild_dist()
                        self.combat_events.append(CombatEvent(
                            'build', u.row, u.col, nr, nc, u.team))
                    worked = True
                    break
            if not worked:
                u.build_timer = 0

    # ── Rules 6–13, 18: combat ───────────────────────────────────────────────

    def _fire(self, attacker: Unit, enemies: list, all_alive: list, wx: dict):
        if attacker.state in (UnitState.SUPPRESSED, UnitState.INHIBITED):
            return
        if attacker.ammo <= 0:
            attacker.morale = max(0.05, attacker.morale - 0.04)
            return

        fire_range = attacker.stats['fire_range']
        vision     = max(3, int(attacker.stats['vision'] * wx['vision_mult']))
        if self.terrain[attacker.row, attacker.col] == URBAN:
            vision = int(vision * 1.25)

        base_dmg   = attacker.stats['damage']
        hit_chance = attacker.stats['hit_chance']

        # Rule 16: comms accuracy penalty
        if not self._has_comms_link(attacker, all_alive):
            hit_chance = max(0.05, hit_chance - self.COMMS_HIT_PENALTY)

        effective_range = min(fire_range, vision)
        closest, min_d = None, effective_range + 1
        for e in enemies:
            if not e.alive:
                continue
            d = max(abs(e.row - attacker.row), abs(e.col - attacker.col))
            if d <= effective_range and d < min_d:
                min_d, closest = d, e

        if closest is None:
            return

        attacker.ammo -= 1
        self.combat_events.append(CombatEvent(
            'shot', attacker.row, attacker.col, closest.row, closest.col, attacker.team))

        # Teamwork bonus (Rule 16 coordination)
        nearby = sum(
            1 for u in all_alive
            if u.alive and u.team == attacker.team and u is not attacker
            and abs(u.row-attacker.row)+abs(u.col-attacker.col) <= 3
        )
        tw_bonus = 1.0 + self.params['teamwork'] * min(nearby, 3) * 0.10

        # Rule 8: fixer — mark target as FIXED
        if closest.state == UnitState.ACTIVE:
            closest.state = UnitState.FIXED
            closest.fixed_timer = max(closest.fixed_timer, self.FIXATION_TICKS)

        # Rule 11: neutralization counter
        closest.hits_taken += 1
        if (closest.hits_taken >= self.NEUTRALIZE_THRESHOLD
                and closest.state != UnitState.INHIBITED):
            closest.state        = UnitState.INHIBITED
            closest.suppress_timer = self.INHIBIT_TICKS
            closest.hits_taken   = 0

        # Hit probability (Rule 25: stochastic uncertainty)
        eff_hit = (hit_chance * attacker.morale
                   * (1.0 - min_d / (fire_range + 1))
                   * tw_bonus
                   * (1.0 - wx['hit_penalty']))

        # Rule 10: ambush from forest
        ambush = 1.0
        if (self.terrain[attacker.row, attacker.col] == FOREST
                and min_d <= 6):
            ambush   = 1.5
            eff_hit  = min(0.95, eff_hit * 1.3)

        if np.random.random() >= eff_hit:
            return  # Miss

        cover  = COVER_BONUS.get(int(self.terrain[closest.row, closest.col]), 0.0)
        damage = base_dmg * ambush * (1.0 - cover) * np.random.uniform(0.7, 1.3)

        # Rule 6: defensive stance halves incoming damage
        if closest.state == UnitState.DEFENSIVE:
            damage *= 0.50

        # Rule 9: breakthrough — 3:1 power ratio increases damage
        local_power = lambda team, row, col: sum(
            u.stats['damage'] for u in all_alive
            if u.team == team and u.alive
            and abs(u.row-row)+abs(u.col-col) <= 4
        )
        att_pow = local_power(attacker.team, closest.row, closest.col)
        def_pow = local_power(closest.team,  closest.row, closest.col)
        if def_pow > 0 and att_pow / def_pow >= 3.0:
            damage *= 1.40

        closest.health -= damage
        closest.morale  = max(0.05, closest.morale - 0.08)
        attacker.morale = min(1.00, attacker.morale + 0.02)

        # Rules 12 & 18: Artillery area suppression
        if attacker.unit_type == UnitType.ARTILLERY:
            area = attacker.stats['area']
            for e in enemies:
                if not e.alive or e is closest:
                    continue
                ad = max(abs(e.row - closest.row), abs(e.col - closest.col))
                if ad <= area:
                    splash = damage * max(0.0, 0.60 - ad * 0.15)
                    e.health -= splash
                    e.morale  = max(0.05, e.morale - 0.15)
                    if e.state == UnitState.ACTIVE:
                        e.state         = UnitState.SUPPRESSED
                        e.suppress_timer = self.SUPPRESSION_TICKS
                    self.combat_events.append(CombatEvent(
                        'suppress', attacker.row, attacker.col, e.row, e.col, attacker.team))

        if closest.health <= 0:
            self.combat_events.append(CombatEvent(
                'kill', attacker.row, attacker.col, closest.row, closest.col, attacker.team))
        else:
            self.combat_events.append(CombatEvent(
                'hit',  attacker.row, attacker.col, closest.row, closest.col, attacker.team))

    # ── Rule 17: logistics resupply ───────────────────────────────────────────

    def _logistics_resupply(self, alive: list):
        for log in alive:
            if log.unit_type != UnitType.LOGISTICS:
                continue
            r = log.stats['supply_range']
            for u in alive:
                if u.team != log.team or u is log:
                    continue
                if max(abs(u.row - log.row), abs(u.col - log.col)) <= r:
                    u.ammo   = min(u.max_ammo,   u.ammo + 3)
                    u.health = min(u.max_health,  u.health + 3.0)
                    u.morale = min(1.0,            u.morale + 0.05)

    # ── Supply depot ──────────────────────────────────────────────────────────

    def _supply_depot(self, alive: list):
        sc = set(map(tuple, np.argwhere(self.terrain == SUPPLY)))
        for u in alive:
            for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                if (u.row+dr, u.col+dc) in sc:
                    u.ammo   = min(u.max_ammo,   u.ammo + 6)
                    u.health = min(u.max_health,  u.health + 5.0)
                    u.morale = min(1.0,            u.morale + 0.08)
                    break

    # ── Rule 21: logistic isolation ───────────────────────────────────────────

    def _isolation_decay(self, alive: list):
        for u in alive:
            rng = self.ISOLATION_RADIUS
            enemies = sum(1 for e in alive if e.team != u.team
                          and max(abs(e.row-u.row), abs(e.col-u.col)) <= rng)
            allies  = sum(1 for a in alive if a.team == u.team and a is not u
                          and max(abs(a.row-u.row), abs(a.col-u.col)) <= rng)
            if enemies >= 4 and allies == 0:
                u.health = max(0.0,  u.health - 3.0)
                u.morale = max(0.05, u.morale - 0.05)

    # ── Rule 20: post-capture reorganization ──────────────────────────────────

    def _check_reorganize(self):
        for u in self.units:
            if (u.alive
                    and u.state == UnitState.ACTIVE
                    and u.team == 0
                    and self.terrain[u.row, u.col] == OBJECTIVE
                    and u.row < self.rows // 2
                    and u.reorg_timer == 0):
                u.state      = UnitState.REORGANIZING
                u.reorg_timer = self.REORG_TICKS

    # ── Grid helpers ──────────────────────────────────────────────────────────

    def _blue_grid(self):
        g = np.zeros((self.rows, self.cols), dtype=float)
        for u in self.units:
            if u.alive and u.team == 0:
                g[u.row, u.col] = u.health / u.max_health
        return g

    def _red_grid(self):
        g = np.zeros((self.rows, self.cols), dtype=float)
        for u in self.units:
            if u.alive and u.team == 1:
                g[u.row, u.col] = u.health / u.max_health
        return g

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

    # ── Winner ────────────────────────────────────────────────────────────────

    def get_winner(self) -> str | None:
        obj_captured = any(
            u.alive and u.team == 0
            and self.terrain[u.row, u.col] == OBJECTIVE
            and u.row < self.rows // 2
            and u.state not in (UnitState.REORGANIZING, UnitState.INACTIVE)
            for u in self.units
        )
        if obj_captured:
            return 'Blue'
        blue_a = sum(1 for u in self.units if u.alive and u.team == 0)
        red_a  = sum(1 for u in self.units if u.alive and u.team == 1)
        if blue_a == 0 and red_a == 0:
            return 'Draw'
        if blue_a == 0:
            return 'Red'
        if red_a == 0:
            return 'Blue'
        return None
