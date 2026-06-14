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

# --- DOCTRINA MILITAR: Estadísticas base por Arma ---
UNIT_STATS = {
    # INFANTERÍA
    UnitType.INFANTRY:       {'hp': 100.0, 'ammo': 30, 'range': 10, 'dmg': 22.0, 'acc': 0.60, 'armor': 0.0, 'vision': 8,  'supply_range': 0},
    # ARTILLERÍA
    UnitType.ARTILLERY:      {'hp': 60.0,  'ammo': 15, 'range': 25, 'dmg': 50.0, 'acc': 0.45, 'armor': 0.0, 'vision': 12, 'supply_range': 0},
    # CABALLERÍA
    UnitType.CAVALRY:        {'hp': 250.0, 'ammo': 40, 'range': 12, 'dmg': 35.0, 'acc': 0.55, 'armor': 0.4, 'vision': 10, 'supply_range': 0},
    # COMUNICACIONES
    UnitType.COMMUNICATIONS: {'hp': 80.0,  'ammo': 20, 'range': 8,  'dmg': 15.0, 'acc': 0.50, 'armor': 0.0, 'vision': 15, 'supply_range': 0},
    # INGENIERÍA
    UnitType.ENGINEERING:    {'hp': 110.0, 'ammo': 25, 'range': 8,  'dmg': 18.0, 'acc': 0.50, 'armor': 0.1, 'vision': 8,  'supply_range': 0},
    # LOGÍSTICA
    UnitType.LOGISTICS:      {'hp': 120.0, 'ammo': 10, 'range': 5,  'dmg': 10.0, 'acc': 0.40, 'armor': 0.1, 'vision': 8,  'supply_range': 3},
}

# ── Unit states ───────────────────────────────────────────────────────────────
class UnitState(IntEnum):
    NORMAL = 0
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
    team:       int       # 0 = Azul, 1 = Rojo
    row:        int
    col:        int
    unit_type:  UnitType  = UnitType.INFANTRY
    state:      UnitState = UnitState.NORMAL
    morale:     float     = 1.0
    health:     float     = -1.0  # Se auto-ajusta al nacer
    ammo:       int       = -1    # Se auto-ajusta al nacer

    def __post_init__(self):
        # Cuando la unidad nace, busca en el diccionario cuánta vida y munición le toca
        if self.health == -1.0:
            self.health = UNIT_STATS[self.unit_type]['hp']
        if self.ammo == -1:
            self.ammo = UNIT_STATS[self.unit_type]['ammo']

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def max_health(self) -> float:
        return UNIT_STATS[self.unit_type]['hp']
        
    @property
    def max_ammo(self) -> int:
        return UNIT_STATS[self.unit_type]['ammo']
    
    @property
    def stats(self) -> dict:
        # Devuelve las estadísticas correspondientes a su tipo de arma
        return UNIT_STATS.get(self.unit_type, UNIT_STATS[UnitType.INFANTRY])


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

    FIRE_RANGE  = 12   # Rango máximo de disparo (distancia Chebyshev)
    BASE_DAMAGE = 22.0 # Daño base de los proyectiles
    HIT_CHANCE  = 0.60 # Probabilidad base de acierto (60%)
    AMMO_MAX    = 30   # Capacidad máxima de munición
    HEALTH_MAX  = 100.0 # Salud máxima de las unidades
    SPAWN_GAP   = 4   # Rule 24: activate if alive < 60% of initial
    ISOLATION_RADIUS = 5

    def __init__(self, terrain: np.ndarray, params: dict = None, weather=None, custom_units=None):
        self.terrain = terrain.copy().astype(int)  # mutable (engineering)
        self.base_terrain = terrain.copy().astype(int)  # original
        self.rows, self.cols = terrain.shape
        self.weather = weather if weather is not None else Weather.CLEAR
        self.params = params or {
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
        blue_start = int(self.rows * 0.52)
        red_start = int(self.rows * 0.22)
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
        seeds = [tuple(p) for p in all_obj if p[0] < self.rows // 2]
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
        # --- HABILIDAD DE LOGÍSTICA: Reparto dinámico ---
        for u in alive:
            if u.unit_type == UnitType.LOGISTICS:
                for ally in alive:
                    if ally.team == u.team and ally != u:
                        if max(abs(ally.row - u.row), abs(ally.col - u.col)) <= 3:
                            ally.ammo = min(ally.max_ammo, ally.ammo + 1)
                            ally.morale = min(1.0, ally.morale + 0.03)

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
        self._supply()

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

    def _move(self, unit, occupied: set, alive=None, wx=None):
        # --- NUEVO: Recuperación natural de valor en cada turno ---
        unit.morale = min(1.0, unit.morale + 0.015)

        # --- 1. REGLAS DEL AUTÓMATA (Cambio de Estado Táctico) ---
        # Si está a punto de morir o su moral colapsa MUCHO -> HUIR
        if unit.health < 20.0 or unit.morale < 0.15:  # Bajamos el umbral para que aguanten más
            unit.state = UnitState.RETREATING
        # Si se queda sin balas -> REORGANIZACIÓN (Buscar base)
        elif unit.ammo <= 0:
            unit.state = UnitState.REORGANIZING
        # Si la moral baja por fuego enemigo -> SUPRESIÓN (Miedo/Cobertura)
        elif unit.morale < 0.50:  # Antes era 0.6, ahora aguantan más antes de esconderse
            unit.state = UnitState.SUPPRESSED
        # Si no tiene órdenes defensivas previas, avanza normal
        elif unit.state != UnitState.DEFENSIVE:
            unit.state = UnitState.NORMAL

        # --- 2. PENALIZACIONES DE MOVIMIENTO (Comportamiento) ---
        # Si está bajo fuego de supresión, hay un 60% de probabilidad de que quede congelado del miedo
        if unit.state == UnitState.SUPPRESSED and np.random.random() > 0.4:
            return
        # Las unidades defensivas rara vez se mueven de su trinchera
        if unit.state == UnitState.DEFENSIVE and np.random.random() > 0.15:
            return
        # Avance normal regido por el optimizador de agresividad
        if unit.state == UnitState.NORMAL and np.random.random() > self.params['aggressiveness']:
            return

        dist_map = self._dist_blue if unit.team == 0 else self._dist_red
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        best_score = -999999.0
        best_pos   = None

        for dr, dc in dirs:
            nr, nc = unit.row + dr, unit.col + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols): continue
            
            t = int(self.terrain[nr, nc])
            if t == OBSTACLE or (nr, nc) in occupied: continue

            d = dist_map[nr, nc]
            if d == np.inf: continue

            cover = COVER_BONUS.get(t, 0.0)
            score = 0.0

            # --- 3. LÓGICA DE DECISIÓN ESPACIAL ---
            if unit.state == UnitState.RETREATING:
                # Regla: Huir del objetivo (maximizar distancia) y buscar bosque/urbano desesperadamente
                score = (d * 5.0) + (cover * 50.0)
                
            elif unit.state == UnitState.REORGANIZING:
                # Regla: Retroceder a zonas seguras para buscar recarga
                score = (d * 3.0) + (cover * 10.0)
                
            elif unit.state == UnitState.SUPPRESSED:
                # Regla: Ignorar el objetivo, priorizar cobertura al 100% (esconderse)
                score = (-d * 0.5) + (cover * 100.0)
                
            else:
                # NORMAL: Avance táctico mezclando ruta óptima (Dijkstra) y cobertura
                score = (-d * self.params['aggressiveness'] * 15.0) + (cover * self.params['cover_seeking'] * 25.0)

            if score > best_score:
                best_score = score
                best_pos   = (nr, nc)

        if best_pos:
            unit.row, unit.col = best_pos   # Cavalry: stop if no valid move

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

    def _fire(self, attacker, enemies: list, alive=None, wx=None):
        stats = UNIT_STATS[attacker.unit_type]
        fire_range = stats['range']
        
        if attacker.ammo <= 0:
            attacker.morale = max(0.05, attacker.morale - 0.04)
            return

        # Buscar objetivo
        closest, min_d = None, fire_range + 1
        for e in enemies:
            if not e.alive: continue
            d = max(abs(e.row - attacker.row), abs(e.col - attacker.col))
            if d < min_d:
                min_d, closest = d, e

        if closest is None: return
        attacker.ammo -= 1

        accuracy_penalty = 0.4 if attacker.state in (UnitState.SUPPRESSED, UnitState.RETREATING) else 1.0

        # --- HABILIDAD DE COMUNICACIONES: Radar táctico ---
        comms_bonus = 1.0
        if alive:
            for ally in alive:
                if ally.team == attacker.team and ally.unit_type == UnitType.COMMUNICATIONS and ally != attacker:
                    if max(abs(ally.row - attacker.row), abs(ally.col - attacker.col)) <= 8:
                        comms_bonus = 1.35  # +35% de puntería si hay un radar cerca
                        break

        nearby_allies = sum(1 for u in self.units if u.alive and u.team == attacker.team and u is not attacker and abs(u.row - attacker.row) + abs(u.col - attacker.col) <= 3)
        teamwork_bonus = 1.0 + self.params['teamwork'] * min(nearby_allies, 3) * 0.1

        hit_prob = (stats['acc'] * attacker.morale * (1.0 - min_d / fire_range) * teamwork_bonus * accuracy_penalty * comms_bonus)

        if np.random.random() < hit_prob:
            # --- HABILIDAD DE CABALLERÍA: Blindaje Pesado ---
            cover = COVER_BONUS.get(int(self.terrain[closest.row, closest.col]), 0.0)
            target_armor = UNIT_STATS[closest.unit_type]['armor']
            
            # El daño final se reduce por la barricada y la armadura del vehículo
            damage_reduction = (1.0 - cover) * (1.0 - target_armor)
            damage = stats['dmg'] * damage_reduction * np.random.uniform(0.7, 1.3)
            
            closest.health -= damage
            closest.morale  = max(0.05, closest.morale - 0.06) # ANTES 0.15: Ahora el daño moral es mucho menor
            
            if closest.morale < 0.50 and closest.state != UnitState.RETREATING:
                closest.state = UnitState.SUPPRESSED

            attacker.morale = min(1.0,  attacker.morale + 0.02)
            if hasattr(self, 'combat_events'):
                self.combat_events.append(CombatEvent(attacker.row, attacker.col, closest.row, closest.col, attacker.team, 'hit'))
        else:
            closest.morale = max(0.05, closest.morale - 0.01) # ANTES 0.05: El ruido de balas fallidas asusta menos
            if hasattr(self, 'combat_events'):
                self.combat_events.append(CombatEvent(attacker.row, attacker.col, closest.row, closest.col, attacker.team, 'shot'))

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

    def _supply(self):
        supply_cells = set(map(tuple, np.argwhere(self.terrain == SUPPLY)))
        if not supply_cells: return
        for u in self.units:
            if not u.alive: continue
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (u.row + dr, u.col + dc) in supply_cells:
                    u.ammo   = min(u.max_ammo,   u.ammo + 6)
                    u.health = min(u.max_health, u.health + 5.0)
                    u.morale = min(1.0,          u.morale + 0.08)
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
