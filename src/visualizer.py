import math
import pygame
import numpy as np
from src.engine import (
    CASimulator, UnitType, UnitState, Weather, WEATHER_NAMES, WEATHER_FX,
    EMPTY, FOREST, OBSTACLE, OBJECTIVE, URBAN, SUPPLY,
    Unit,
)

# ── Terrain palette ───────────────────────────────────────────────────────────
TERRAIN_BASE = {
    EMPTY:     (18,  22,  28),
    FOREST:    (32, 105,  38),
    OBSTACLE:  (42,  88, 138),
    OBJECTIVE: (155, 22,  22),
    URBAN:     (138, 130, 102),
    SUPPLY:    (185, 165,  22),
}

# ── UI colours ────────────────────────────────────────────────────────────────
BG_COLOR      = (10,  14,  20)
PANEL_BG      = (20,  25,  32)
PANEL_BORDER  = (46,  52,  60)
TEXT_WHITE    = (228, 228, 228)
TEXT_DIM      = (130, 130, 130)
GOLD          = (255, 210,   0)
GREEN_HP      = (55,  195,  75)
YELLOW_AMMO   = (225, 195,  45)
BLUE_BRIGHT   = (55,  155, 255)
RED_BRIGHT    = (255,  62,  62)

# ── Unit body colours — 6 arms × 2 teams ─────────────────────────────────────
UNIT_BODY = {
    (0, UnitType.INFANTRY):       ( 60, 140, 255),  # Blue infantry
    (0, UnitType.ARTILLERY):      ( 55, 215, 170),  # Blue artillery  — teal
    (0, UnitType.CAVALRY):        ( 28,  75, 195),  # Blue cavalry    — navy
    (0, UnitType.COMMUNICATIONS): ( 90, 220, 230),  # Blue comms      — cyan
    (0, UnitType.ENGINEERING):    (115, 180,  80),  # Blue engineering— olive
    (0, UnitType.LOGISTICS):      (200, 190,  50),  # Blue logistics  — gold
    (1, UnitType.INFANTRY):       (255,  60,  60),  # Red infantry
    (1, UnitType.ARTILLERY):      (255, 135,  55),  # Red artillery   — orange
    (1, UnitType.CAVALRY):        (175,  28,  28),  # Red cavalry     — dark red
    (1, UnitType.COMMUNICATIONS): (220,  80, 200),  # Red comms       — magenta
    (1, UnitType.ENGINEERING):    (180, 130,  40),  # Red engineering — brown
    (1, UnitType.LOGISTICS):      (200, 170,  90),  # Red logistics   — tan
}
HELMET_BLUE = (25,  55, 135)
HELMET_RED  = (115, 18,  18)
SKIN_COLOR  = (218, 175, 135)
METAL_COLOR = (155, 155, 155)

# State overlay colours
STATE_OVERLAY = {
    UnitState.SUPPRESSED:   (255, 220,   0, 160),  # yellow flash
    UnitState.FIXED:        (255, 100,   0, 140),  # orange
    UnitState.INHIBITED:    (200,   0, 200, 140),  # magenta
    UnitState.RETREATING:   (255, 255, 255, 120),  # white
    UnitState.REORGANIZING: (  0, 220, 100, 130),  # green
    UnitState.DEFENSIVE:    (  0, 160, 255, 110),  # blue tint
}

# Arm labels for UI
ARM_LABELS = {
    UnitType.INFANTRY:       'INF',
    UnitType.ARTILLERY:      'ART',
    UnitType.CAVALRY:        'CAB',
    UnitType.COMMUNICATIONS: 'COM',
    UnitType.ENGINEERING:    'ING',
    UnitType.LOGISTICS:      'LOG',
}

ARM_FULL = {
    UnitType.INFANTRY:       'Infanteria',
    UnitType.ARTILLERY:      'Artilleria',
    UnitType.CAVALRY:        'Caballeria',
    UnitType.COMMUNICATIONS: 'Comunicaciones',
    UnitType.ENGINEERING:    'Ingenieria',
    UnitType.LOGISTICS:      'Logistica',
}

WEATHER_ICON = {
    Weather.CLEAR: 'SOL',
    Weather.RAIN:  'LLU',
    Weather.FOG:   'NIE',
    Weather.COLD:  'FRI',
    Weather.HEAT:  'CAL',
}
WEATHER_COLOR = {
    Weather.CLEAR: (255, 230,  60),
    Weather.RAIN:  ( 80, 140, 255),
    Weather.FOG:   (170, 170, 200),
    Weather.COLD:  (140, 200, 255),
    Weather.HEAT:  (255, 120,  40),
}


# ── Particle ──────────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size')

    def __init__(self, x, y, vx, vy, life, color, size=2):
        self.x, self.y       = float(x), float(y)
        self.vx, self.vy     = float(vx), float(vy)
        self.life = self.max_life = float(life)
        self.color = color
        self.size  = size

    def update(self, dt: float):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += 55 * dt   # gravity
        self.life -= dt

    @property
    def alive(self): return self.life > 0

    @property
    def alpha(self): return max(0.0, self.life / self.max_life)


# ── Tracer (bullet line) ──────────────────────────────────────────────────────
class Tracer:
    __slots__ = ('x1', 'y1', 'x2', 'y2', 'life', 'color', 'width')

    def __init__(self, x1, y1, x2, y2, color, width=1):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.life  = 0.18
        self.color = color
        self.width = width

    def update(self, dt: float): self.life -= dt

    @property
    def alive(self): return self.life > 0


# ── Main visualizer ───────────────────────────────────────────────────────────
class PygameVisualizer:
    """WorldBox-inspired real-time Pygame visualizer for the military CA sim."""

    PANEL_WIDTH   = 300
    MIN_CELL      = 5
    MAX_CELL      = 18
    DEFAULT_CELL  = 10
    ANIM_INTERVAL = 0.20   # seconds per walk frame

    def __init__(self, terrain: np.ndarray, sim: CASimulator,
                 fps: int = 8, title: str = 'Military CA Simulation'):
        self.terrain = terrain
        self.sim     = sim
        self.rows, self.cols = terrain.shape
        self.fps   = fps
        self.title = title

        pygame.init()
        info  = pygame.display.Info()
        max_w = info.current_w - 80 - self.PANEL_WIDTH
        max_h = info.current_h - 80
        cw    = max_w // self.cols
        ch    = max_h // self.rows
        self.cell = max(self.MIN_CELL, min(self.MAX_CELL, min(cw, ch)))

        self.map_w = self.cols * self.cell
        self.map_h = self.rows * self.cell
        self.win_w = self.map_w + self.PANEL_WIDTH
        self.win_h = self.map_h

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(title)

        self.terrain_surf = self._bake_terrain()

        self.particles: list[Particle] = []
        self.tracers:   list[Tracer]   = []

        self.paused  = False
        self.running = True
        self.step    = 0
        self.winner  = None
        self.clock   = pygame.time.Clock()

        self.anim_frame = 0
        self.anim_timer = 0.0

        # ── Heat map ──────────────────────────────────────────────────────
        self.heat_map    = np.zeros((self.rows, self.cols), dtype=float)
        self.show_heatmap = False

        # ── Battle chart history ──────────────────────────────────────────
        self.blue_history: list[int] = []
        self.red_history:  list[int] = []
        self.max_units = 1   # set on first step

        self.font_big   = pygame.font.SysFont('Consolas', 22, bold=True)
        self.font_med   = pygame.font.SysFont('Consolas', 15, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 12)
        self.font_tiny  = pygame.font.SysFont('Consolas', 10)

    # ── Terrain baking ────────────────────────────────────────────────────────

    def _bake_terrain(self) -> pygame.Surface:
        """Pre-render all terrain tiles once with pixel-art texture."""
        surf = pygame.Surface((self.map_w, self.map_h))
        rng  = np.random.default_rng(7)          # fixed seed → stable look
        c    = self.cell

        for r in range(self.rows):
            for col in range(self.cols):
                t    = int(self.terrain[r, col])
                base = TERRAIN_BASE.get(t, TERRAIN_BASE[EMPTY])
                x0   = col * c
                y0   = r   * c

                # Base tile with per-cell hue variation
                var  = rng.integers(-14, 15, size=3)
                fill = tuple(max(0, min(255, base[i] + int(var[i]))) for i in range(3))
                pygame.draw.rect(surf, fill, (x0, y0, c, c))

                # ── Per-terrain texture ──────────────────────────────────────
                if t == FOREST:
                    # Scattered leaf highlights
                    for _ in range(max(1, c // 3)):
                        px = x0 + int(rng.integers(0, c))
                        py = y0 + int(rng.integers(0, c))
                        leaf = (max(0, base[0]-18), min(255, base[1]+35), max(0, base[2]-8))
                        surf.set_at((px, py), leaf)
                    # Occasional darker trunk pixel in centre
                    if c >= 7 and rng.integers(0, 3) == 0:
                        trunk = (max(0, base[0]-30), max(0, base[1]-20), max(0, base[2]-10))
                        surf.set_at((x0 + c//2, y0 + c//2), trunk)

                elif t == OBSTACLE:
                    # Horizontal water-shimmer lines
                    for ly in range(0, c, max(2, c//4)):
                        wave = (max(0, base[0]-10), min(255, base[1]+18), min(255, base[2]+22))
                        pygame.draw.line(surf, wave, (x0, y0+ly), (x0+c-1, y0+ly))

                elif t == OBJECTIVE:
                    # Bright cross marker in the centre
                    if c >= 7:
                        mx, my = x0 + c//2, y0 + c//2
                        mark = (min(255, base[0]+80), base[1]+10, base[2]+10)
                        pygame.draw.line(surf, mark, (mx-2, my), (mx+2, my))
                        pygame.draw.line(surf, mark, (mx, my-2), (mx, my+2))

                elif t == URBAN:
                    # Subtle grid-line border
                    border = tuple(max(0, base[i]-28) for i in range(3))
                    pygame.draw.rect(surf, border, (x0, y0, c, c), 1)

                elif t == SUPPLY:
                    # Small cross / plus in centre
                    if c >= 6:
                        mx, my = x0 + c//2, y0 + c//2
                        bright = tuple(min(255, base[i]+50) for i in range(3))
                        pygame.draw.line(surf, bright, (mx-2, my), (mx+2, my), 2)
                        pygame.draw.line(surf, bright, (mx, my-2), (mx, my+2), 2)

        return surf

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _center(self, row, col):
        c = self.cell
        return int(col * c + c / 2), int(row * c + c / 2)

    # ── Particle spawners ─────────────────────────────────────────────────────

    def _spawn_explosion(self, row, col, count=18):
        cx, cy = self._center(row, col)
        fire_palette = [(255, 100, 25), (255, 200, 45), (200, 55, 8), (255, 255, 90)]
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(45, 140)
            life  = np.random.uniform(0.4, 1.0)
            col_p = fire_palette[np.random.randint(len(fire_palette))]
            self.particles.append(Particle(
                cx, cy,
                math.cos(angle)*speed, math.sin(angle)*speed,
                life, col_p, np.random.randint(1, 4)
            ))
        # Smoke puffs
        for _ in range(7):
            angle = np.random.uniform(0, 2*math.pi)
            speed = np.random.uniform(8, 35)
            life  = np.random.uniform(0.7, 1.4)
            self.particles.append(Particle(
                cx, cy,
                math.cos(angle)*speed, math.sin(angle)*speed - 22,
                life, (95, 95, 95), 3
            ))

    def _spawn_sparks(self, row, col, count=7):
        cx, cy = self._center(row, col)
        for _ in range(count):
            angle = np.random.uniform(0, 2*math.pi)
            speed = np.random.uniform(22, 75)
            life  = np.random.uniform(0.10, 0.28)
            self.particles.append(Particle(
                cx, cy,
                math.cos(angle)*speed, math.sin(angle)*speed,
                life, (255, 215, 75), 2
            ))

    def _process_combat_events(self, events: list):
        for ev in events:
            x1, y1 = self._center(ev.src_row, ev.src_col)
            x2, y2 = self._center(ev.dst_row, ev.dst_col)
            tc = (90, 195, 255) if ev.team == 0 else (255, 95, 75)

            if ev.event_type == 'shot':
                self.tracers.append(Tracer(x1, y1, x2, y2, tc, 1))
                # Shooter position heats up slightly
                self.heat_map[ev.src_row, ev.src_col] += 0.3
            elif ev.event_type == 'hit':
                self.tracers.append(Tracer(x1, y1, x2, y2, tc, 1))
                self._spawn_sparks(ev.dst_row, ev.dst_col)
                self.heat_map[ev.dst_row, ev.dst_col] += 1.0
            elif ev.event_type == 'kill':
                self.tracers.append(Tracer(x1, y1, x2, y2, tc, 2))
                self._spawn_explosion(ev.dst_row, ev.dst_col)
                # Deaths leave a strong mark on the heat map
                self.heat_map[ev.dst_row, ev.dst_col] += 4.0

    # ── Pixel-art sprite drawing ──────────────────────────────────────────────

    def _draw_infantry(self, surf, cx, cy, body, helmet, anim):
        """Humanoid soldier: helmet + skin face + body + animated arms & legs."""
        c    = self.cell
        half = c // 2

        if c >= 8:
            hr = max(2, c // 5)          # head radius
            hy = cy - half + hr + 1      # head centre y

            # Helmet (slightly larger, darker)
            pygame.draw.circle(surf, helmet, (cx, hy), hr + 1)
            # Face / skin
            pygame.draw.circle(surf, SKIN_COLOR, (cx, hy), hr)
            # Tiny eye dots
            if c >= 10:
                pygame.draw.circle(surf, (30, 20, 10), (cx - 1, hy), 1)
                pygame.draw.circle(surf, (30, 20, 10), (cx + 1, hy), 1)

            # Torso
            t_top = hy + hr
            t_h   = max(3, half // 2 + 1)
            pygame.draw.rect(surf, body, (cx - 2, t_top, 5, t_h))

            # Arms (alternate per anim frame)
            ay = t_top + 1
            if anim == 0:
                pygame.draw.line(surf, body, (cx - 2, ay), (cx - 4, ay + 2), 2)
                pygame.draw.line(surf, body, (cx + 2, ay), (cx + 4, ay + 1), 2)
            else:
                pygame.draw.line(surf, body, (cx - 2, ay), (cx - 4, ay + 1), 2)
                pygame.draw.line(surf, body, (cx + 2, ay), (cx + 4, ay + 2), 2)

            # Legs (walk cycle)
            lt  = t_top + t_h
            lh  = max(2, c // 4)
            if anim == 0:
                pygame.draw.rect(surf, body, (cx - 3, lt,     2, lh))
                pygame.draw.rect(surf, body, (cx + 1, lt + 1, 2, lh - 1))
            else:
                pygame.draw.rect(surf, body, (cx - 3, lt + 1, 2, lh - 1))
                pygame.draw.rect(surf, body, (cx + 1, lt,     2, lh))
        else:
            # Tiny cells → simple coloured circle with helmet top
            pygame.draw.circle(surf, helmet, (cx, cy), half)
            pygame.draw.circle(surf, body,   (cx, cy + 1), half - 1)

    def _draw_sniper(self, surf, cx, cy, body, helmet, anim):
        """Crouching sniper with extended rifle barrel."""
        c    = self.cell
        half = c // 2

        if c >= 8:
            hr  = max(2, c // 6)
            hy  = cy - half + hr + 2
            pygame.draw.circle(surf, helmet, (cx, hy), hr + 1)
            pygame.draw.circle(surf, SKIN_COLOR, (cx, hy), hr)

            # Compact crouching torso
            t_top = hy + hr
            t_h   = max(2, half // 3)
            pygame.draw.rect(surf, body, (cx - 1, t_top, 3, t_h))

            # Rifle barrel (grey, extending right)
            by_ = t_top + 1
            pygame.draw.line(surf, METAL_COLOR, (cx + 1, by_), (cx + half + 4, by_), 2)
            # Scope dot
            pygame.draw.circle(surf, (40, 40, 40), (cx + half + 4, by_), 1)

            # Crouched legs (short)
            lt  = t_top + t_h
            lh  = max(1, c // 5)
            pygame.draw.rect(surf, body, (cx - 2, lt, 2, lh + anim))
            pygame.draw.rect(surf, body, (cx + 1, lt, 2, lh + (1 - anim)))
        else:
            # Diamond for small cells
            pts = [(cx, cy-half), (cx+half, cy), (cx, cy+half), (cx-half, cy)]
            pygame.draw.polygon(surf, body, pts)
            pygame.draw.polygon(surf, helmet, pts, 1)

    def _draw_artillery(self, surf, cx, cy, body, anim):
        """Artillery: wide base + long upward barrel."""
        c    = self.cell
        half = c // 2
        if c >= 7:
            # Base platform
            pygame.draw.rect(surf, body, (cx - half + 1, cy, c - 2, half - 1))
            pygame.draw.line(surf, METAL_COLOR, (cx - half + 2, cy), (cx + half - 2, cy))
            # Barrel (points upward, slight left lean each frame)
            bx = cx - anim
            pygame.draw.line(surf, METAL_COLOR, (cx, cy), (bx, cy - half - 3), 3)
            pygame.draw.circle(surf, (60, 60, 60), (bx, cy - half - 3), 2)
            # Wheels
            pygame.draw.circle(surf, (50, 50, 50), (cx - half + 3, cy + half - 2), 3)
            pygame.draw.circle(surf, (50, 50, 50), (cx + half - 3, cy + half - 2), 3)
        else:
            # Small: pentagon
            pts = [(cx, cy-half), (cx+half, cy-half//2), (cx+half//2, cy+half),
                   (cx-half//2, cy+half), (cx-half, cy-half//2)]
            pygame.draw.polygon(surf, body, pts)

    def _draw_cavalry(self, surf, cx, cy, body, helmet, anim):
        """Cavalry: rider on horse silhouette."""
        c    = self.cell
        half = c // 2
        if c >= 8:
            horse_col = tuple(max(0, v - 30) for v in body)
            # Horse body
            pygame.draw.ellipse(surf, horse_col,
                                (cx - half, cy + 1, c, half - 1))
            # Legs (animated)
            leg_y = cy + half
            offsets = [(-3, 0), (-1, 0), (1, 0), (3, 0)]
            for i, (ox, _) in enumerate(offsets):
                stride = (anim + i) % 2
                pygame.draw.line(surf, horse_col,
                                 (cx + ox, leg_y), (cx + ox, leg_y + 3 - stride), 2)
            # Rider (small)
            hr = max(2, c // 6)
            hy = cy - hr
            pygame.draw.circle(surf, helmet,     (cx, hy), hr + 1)
            pygame.draw.circle(surf, SKIN_COLOR, (cx, hy), hr)
            pygame.draw.rect(surf, body, (cx - 2, hy + hr, 4, half - 1))
        else:
            pygame.draw.ellipse(surf, body, (cx - half + 1, cy - 2, c - 2, half + 2))

    def _draw_communications(self, surf, cx, cy, body, anim):
        """Comms: small body + tall antenna with blinking tip."""
        c    = self.cell
        half = c // 2
        if c >= 7:
            # Box body
            pygame.draw.rect(surf, body, (cx - 3, cy - 1, 7, half))
            # Antenna mast
            pygame.draw.line(surf, METAL_COLOR, (cx, cy - 1), (cx, cy - half - 2), 2)
            # Dish arms
            pygame.draw.line(surf, METAL_COLOR, (cx, cy - half + 1),
                             (cx - 4, cy - half - 1), 1)
            pygame.draw.line(surf, METAL_COLOR, (cx, cy - half + 1),
                             (cx + 4, cy - half - 1), 1)
            # Blinking signal tip
            tip_col = (  0, 255, 100) if anim == 0 else (0, 120, 60)
            pygame.draw.circle(surf, tip_col, (cx, cy - half - 2), 2)
        else:
            pts = [(cx, cy-half), (cx+half, cy), (cx, cy+half), (cx-half, cy)]
            pygame.draw.polygon(surf, body, pts)

    def _draw_engineering(self, surf, cx, cy, body, anim):
        """Engineering: body + wrench/tool cross."""
        c    = self.cell
        half = c // 2
        if c >= 7:
            pygame.draw.rect(surf, body, (cx - 3, cy - 2, 7, half + 2))
            # Tool cross (wrench suggestion)
            tool = tuple(min(255, v + 40) for v in body)
            pygame.draw.line(surf, tool, (cx - half + 2, cy + 2),
                             (cx + half - 2, cy + 2), 2)
            pygame.draw.line(surf, tool, (cx, cy - half + 2),
                             (cx, cy + 2), 2)
            # Animated gear dot
            angle = anim * math.pi
            gx = cx + int(3 * math.cos(angle))
            gy = cy + 2 + int(3 * math.sin(angle))
            pygame.draw.circle(surf, METAL_COLOR, (gx, gy), 1)
        else:
            pygame.draw.rect(surf, body, (cx - half + 1, cy - half + 1, c - 2, c - 2))
            pygame.draw.line(surf, METAL_COLOR, (cx - half + 2, cy), (cx + half - 2, cy), 1)
            pygame.draw.line(surf, METAL_COLOR, (cx, cy - half + 2), (cx, cy + half - 2), 1)

    def _draw_logistics(self, surf, cx, cy, body, anim):
        """Logistics: truck/box with wheels."""
        c    = self.cell
        half = c // 2
        if c >= 7:
            # Cargo box
            cargo = tuple(min(255, v + 20) for v in body)
            pygame.draw.rect(surf, cargo, (cx - half + 1, cy - half + 2, c - 2, half + 1))
            pygame.draw.rect(surf, body, (cx - half + 1, cy - half + 2, c - 2, half + 1), 1)
            # Cab
            pygame.draw.rect(surf, body, (cx + 1, cy - half + 1, half - 1, half))
            # Wheels (animated roll)
            woff = anim * 1
            for wx2 in [cx - half + 3, cx + half - 3]:
                pygame.draw.circle(surf, (40, 40, 40), (wx2 + woff, cy + 3), 3)
                pygame.draw.circle(surf, (90, 90, 90), (wx2 + woff, cy + 3), 1)
        else:
            pygame.draw.rect(surf, body, (cx - half + 1, cy - half + 1, c - 2, c - 2))

    def _draw_tank(self, surf, cx, cy, body, anim):
        """Chunky tank with hull, tracks, turret and rotating barrel."""
        c    = self.cell
        half = c // 2

        if c >= 7:
            hw   = c - 2           # hull width
            hh   = max(4, half)    # hull height
            hx   = cx - hw // 2
            hy   = cy - hh // 2 + 2

            # Shadow
            dark = tuple(max(0, v - 45) for v in body)
            pygame.draw.rect(surf, dark, (hx - 1, hy + 1, hw + 2, hh))
            # Hull
            pygame.draw.rect(surf, body, (hx, hy, hw, hh))
            # Hull highlight (top edge)
            light = tuple(min(255, v + 35) for v in body)
            pygame.draw.line(surf, light, (hx, hy), (hx + hw - 1, hy))

            # Tracks (dark strip at bottom with animated segments)
            track = (35, 35, 35)
            pygame.draw.rect(surf, track, (hx, hy + hh - 2, hw, 3))
            seg_count = hw // 3
            for i in range(seg_count + 1):
                sx = hx + (i * 3 + anim * 2) % hw
                pygame.draw.line(surf, (65, 65, 65),
                                 (sx, hy + hh - 2), (sx, hy + hh))

            # Turret
            tw    = max(4, hw // 2)
            th    = max(3, hh // 2)
            tx    = cx - tw // 2
            ty    = hy - th + 2
            t_col = tuple(min(255, v + 22) for v in body)
            pygame.draw.rect(surf, t_col, (tx, ty, tw, th))
            pygame.draw.line(surf, light, (tx, ty), (tx + tw - 1, ty))

            # Barrel
            pygame.draw.rect(surf, METAL_COLOR,
                             (cx, ty + th // 2 - 1, half + 4, 2))
            pygame.draw.circle(surf, (80, 80, 80), (cx + half + 4, ty + th // 2), 1)
        else:
            pygame.draw.rect(surf, body, (cx - half + 1, cy - half + 1, c - 2, c - 2))

    # ── Heat map overlay ──────────────────────────────────────────────────────

    def _draw_heatmap(self, surface):
        """Render a colour-coded overlay showing where combat was most intense."""
        peak = self.heat_map.max()
        if peak == 0:
            return

        norm = self.heat_map / peak
        c    = self.cell
        hm   = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

        for r in range(self.rows):
            for col in range(self.cols):
                v = float(norm[r, col])
                if v < 0.04:
                    continue
                # Gradient: low → yellow, mid → orange, high → red
                red_c   = min(255, int(180 + v * 75))
                green_c = max(0,   int(210 - v * 210))
                alpha   = min(210, int(v * 230))
                hm.fill((red_c, green_c, 0, alpha),
                        (col * c, r * c, c, c))

        surface.blit(hm, (0, 0))

        # Label on map so the viewer knows what they're seeing
        label = self.font_med.render('MAPA DE CALOR  [H para salir]',
                                     True, (255, 230, 80))
        surface.blit(label, (6, 4))

    # ── Live battle chart ─────────────────────────────────────────────────────

    def _draw_battle_chart(self, surface, x: int, y: int,
                           width: int, height: int):
        """Mini line chart: Blue vs Red unit count over time."""
        # Background
        pygame.draw.rect(surface, (12, 16, 22), (x, y, width, height),
                         border_radius=4)
        pygame.draw.rect(surface, PANEL_BORDER, (x, y, width, height),
                         1, border_radius=4)

        n = len(self.blue_history)
        if n < 2:
            hint = self.font_tiny.render('Esperando datos...', True, TEXT_DIM)
            surface.blit(hint, (x + 6, y + height // 2 - 6))
            return

        mx = max(self.max_units, 1)
        pad_x, pad_y = 4, 4

        def to_px(i, val):
            sx = x + pad_x + int(i * (width  - pad_x * 2) / (n - 1))
            sy = y + height - pad_y - int(val * (height - pad_y * 2) / mx)
            return sx, sy

        # Zero line (light)
        zy = y + height - pad_y
        pygame.draw.line(surface, PANEL_BORDER, (x + pad_x, zy),
                         (x + width - pad_x, zy))

        # Fill areas under each line (translucent)
        for history, base_color in [
            (self.blue_history, (40, 100, 200, 60)),
            (self.red_history,  (200, 40,  40, 60)),
        ]:
            fill = pygame.Surface((width, height), pygame.SRCALPHA)
            pts  = [to_px(i, v) for i, v in enumerate(history)]
            # Close polygon to zero line
            poly = pts + [(pts[-1][0], zy), (pts[0][0], zy)]
            # Shift pts to local fill surface coords
            local_poly = [(px - x, py - y) for px, py in poly]
            if len(local_poly) >= 3:
                pygame.draw.polygon(fill, base_color, local_poly)
            surface.blit(fill, (x, y))

        # Lines on top
        blue_pts = [to_px(i, v) for i, v in enumerate(self.blue_history)]
        red_pts  = [to_px(i, v) for i, v in enumerate(self.red_history)]
        pygame.draw.lines(surface, (60, 140, 255), False, blue_pts, 2)
        pygame.draw.lines(surface, RED_BRIGHT,     False, red_pts,  2)

        # Current values at right edge
        bv = self.blue_history[-1]
        rv = self.red_history[-1]
        surface.blit(self.font_tiny.render(str(bv), True, (60, 140, 255)),
                     (x + width - 24, blue_pts[-1][1] - 8))
        surface.blit(self.font_tiny.render(str(rv), True, RED_BRIGHT),
                     (x + width - 24, red_pts[-1][1]  + 2))

    # ── Draw all units ────────────────────────────────────────────────────────

    def _draw_units(self, surface):
        half  = self.cell // 2
        ticks = pygame.time.get_ticks()

        for u in self.sim.units:
            if not u.alive:
                continue

            cx, cy = self._center(u.row, u.col)
            body   = UNIT_BODY.get((u.team, u.unit_type), (128, 128, 128))
            helmet = HELMET_BLUE if u.team == 0 else HELMET_RED

            # Low-HP white blink
            if u.health < u.max_health * 0.25 and (ticks // 160) % 2 == 0:
                body   = (255, 255, 255)
                helmet = (210, 210, 210)

            if u.unit_type == UnitType.INFANTRY:
                self._draw_infantry(surface, cx, cy, body, helmet, self.anim_frame)
            elif u.unit_type == UnitType.ARTILLERY:
                self._draw_artillery(surface, cx, cy, body, self.anim_frame)
            elif u.unit_type == UnitType.CAVALRY:
                self._draw_cavalry(surface, cx, cy, body, helmet, self.anim_frame)
            elif u.unit_type == UnitType.COMMUNICATIONS:
                self._draw_communications(surface, cx, cy, body, self.anim_frame)
            elif u.unit_type == UnitType.ENGINEERING:
                self._draw_engineering(surface, cx, cy, body, self.anim_frame)
            elif u.unit_type == UnitType.LOGISTICS:
                self._draw_logistics(surface, cx, cy, body, self.anim_frame)

            # State overlay (coloured ring / flash)
            ov = STATE_OVERLAY.get(u.state)
            if ov:
                blink_states = (UnitState.SUPPRESSED, UnitState.INHIBITED)
                if u.state not in blink_states or (ticks // 200) % 2 == 0:
                    ov_surf = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
                    pygame.draw.rect(ov_surf, ov, (0, 0, self.cell, self.cell),
                                     border_radius=2)
                    surface.blit(ov_surf, (cx - half, cy - half))

            # HP bar below unit
            if u.health < u.max_health * 0.99:
                bw = self.cell
                bx = cx - bw // 2
                by = cy + half + 1
                hp_ratio = max(0.0, u.health / u.max_health)
                pygame.draw.rect(surface, (50, 14, 14), (bx, by, bw, 2))
                bar_col = (GREEN_HP if hp_ratio > 0.5
                           else (255, 160, 0) if hp_ratio > 0.25
                           else RED_BRIGHT)
                pygame.draw.rect(surface, bar_col,
                                 (bx, by, max(1, int(bw * hp_ratio)), 2))

    # ── Visual effects ────────────────────────────────────────────────────────

    def _draw_fx(self, surface, dt: float):
        # Tracers
        for t in self.tracers:
            # Dim as they fade
            fade  = max(0.0, t.life / 0.18)
            color = tuple(max(0, min(255, int(v * fade))) for v in t.color)
            pygame.draw.line(surface, color,
                             (t.x1, t.y1), (t.x2, t.y2), t.width)
            t.update(dt)
        self.tracers = [t for t in self.tracers if t.alive]

        # Particles
        for p in self.particles:
            a     = p.alpha
            color = tuple(max(0, min(255, int(v * a))) for v in p.color)
            px, py = int(p.x), int(p.y)
            if p.size <= 1:
                if 0 <= px < self.map_w and 0 <= py < self.map_h:
                    surface.set_at((px, py), color)
            else:
                pygame.draw.circle(surface, color, (px, py), p.size)
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    # ── Fog of war ────────────────────────────────────────────────────────────

    def _draw_fog(self, surface):
        fog = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        fog.fill((0, 0, 0, 148))
        for u in self.sim.units:
            if not u.alive or u.team != 0:
                continue
            cx, cy = self._center(u.row, u.col)
            pygame.draw.circle(fog, (0, 0, 0, 0),
                               (cx, cy), u.stats['vision'] * self.cell)
        surface.blit(fog, (0, 0))

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _draw_panel(self, surface, stats):
        px = self.map_w
        pygame.draw.rect(surface, PANEL_BG, (px, 0, self.PANEL_WIDTH, self.win_h))
        pygame.draw.line(surface, PANEL_BORDER, (px, 0), (px, self.win_h), 2)

        x    = px + 12
        y    = 14
        bw   = 250   # usable bar / content width

        # ── Step counter + speed ──────────────────────────────────────────
        surface.blit(self.font_big.render(f'PASO {self.step:03d}', True, BLUE_BRIGHT), (x, y))
        spd_txt = 'EN PAUSA' if self.paused else f'{self.fps} FPS'
        spd_col = GOLD if self.paused else TEXT_DIM
        spd_surf = self.font_small.render(spd_txt, True, spd_col)
        surface.blit(spd_surf, (x + bw - spd_surf.get_width(), y + 5))
        y += 30

        pygame.draw.line(surface, PANEL_BORDER, (x, y), (x + bw, y), 1)
        y += 8

        # ── Helpers ───────────────────────────────────────────────────────

        def draw_bar(lbl, ratio, bar_col, val_str):
            nonlocal y
            lbl_surf = self.font_tiny.render(lbl, True, TEXT_DIM)
            surface.blit(lbl_surf, (x, y))
            lbl_w = lbl_surf.get_width() + 4
            bar_x = x + lbl_w
            bar_w2 = bw - lbl_w - 32
            pygame.draw.rect(surface, (38, 38, 38), (bar_x, y + 1, bar_w2, 6), border_radius=3)
            if ratio > 0:
                pygame.draw.rect(surface, bar_col,
                                 (bar_x, y + 1, max(2, int(bar_w2 * ratio)), 6),
                                 border_radius=3)
            val_surf = self.font_tiny.render(val_str, True, bar_col)
            surface.blit(val_surf, (bar_x + bar_w2 + 3, y))
            y += 11

        def draw_unit_type_row(team_idx):
            """Draw all 6 arm counts in 2 rows of 3."""
            nonlocal y
            counts = {ut: 0 for ut in UnitType}
            for u in self.sim.units:
                if u.team == team_idx:
                    counts[u.unit_type] += 1

            arm_col = lambda ut: UNIT_BODY.get((team_idx, ut), (128, 128, 128))
            specs = list(UnitType)
            col_w  = bw // 3

            for row_idx in range(2):
                for col_idx in range(3):
                    ut  = specs[row_idx * 3 + col_idx]
                    col = arm_col(ut)
                    ix  = x + col_idx * col_w
                    iy  = y + 7
                    pygame.draw.rect(surface, col, (ix, iy - 4, 8, 8))
                    pygame.draw.rect(surface, TEXT_WHITE, (ix, iy - 4, 8, 8), 1)
                    lbl = f' {ARM_LABELS[ut]}:{counts[ut]}'
                    surface.blit(self.font_tiny.render(lbl, True, col), (ix + 10, y + 1))
                y += 14

        def team_block(prefix, label, label_color, team_idx):
            nonlocal y
            alive = stats[f'{prefix}_alive']
            title_surf = self.font_med.render(label, True, label_color)
            surface.blit(title_surf, (x, y))
            alive_surf = self.font_small.render(f'vivos:{alive}', True, TEXT_WHITE)
            surface.blit(alive_surf, (x + bw - alive_surf.get_width(), y + 2))
            y += 18

            draw_unit_type_row(team_idx)

            hp     = stats[f'{prefix}_avg_health']
            ammo   = stats[f'{prefix}_avg_ammo']
            morale = stats[f'{prefix}_avg_morale']
            hp_col = (GREEN_HP if hp > 80
                      else (255, 160, 0) if hp > 40
                      else RED_BRIGHT)
            draw_bar('Salud   ', min(1.0, hp / 200.0), hp_col,        f'{hp:.0f}')
            draw_bar('Municion', min(1.0, ammo / 30.0), YELLOW_AMMO,  f'{ammo:.1f}')
            draw_bar('Moral   ', min(1.0, morale),      (95, 195, 255), f'{morale:.2f}')
            y += 4

        # ── Weather indicator ─────────────────────────────────────────────
        wx_col  = WEATHER_COLOR.get(self.sim.weather, TEXT_WHITE)
        wx_icon = WEATHER_ICON.get(self.sim.weather, '?')
        wx_name = WEATHER_NAMES.get(self.sim.weather, '')
        wx_surf = self.font_med.render(f'[{wx_icon}] {wx_name}', True, wx_col)
        surface.blit(wx_surf, (x, y))
        y += 18
        pygame.draw.line(surface, PANEL_BORDER, (x, y), (x + bw, y), 1)
        y += 8

        # ── Blue team ─────────────────────────────────────────────────────
        team_block('blue', 'AZUL (ATACANTE)', (60, 140, 255), 0)
        pygame.draw.line(surface, PANEL_BORDER, (x, y), (x + bw, y), 1)
        y += 8

        # ── Red team ──────────────────────────────────────────────────────
        team_block('red', 'ROJO (DEFENSOR)', RED_BRIGHT, 1)
        pygame.draw.line(surface, PANEL_BORDER, (x, y), (x + bw, y), 1)
        y += 8

        # ── Live battle chart ──────────────────────────────────────────────
        surface.blit(self.font_med.render('BAJAS EN TIEMPO REAL', True, TEXT_WHITE), (x, y))
        y += 16
        self._draw_battle_chart(surface, x, y, bw, 64)
        y += 70
        pygame.draw.line(surface, PANEL_BORDER, (x, y), (x + bw, y), 1)
        y += 8

        # ── Terrain legend ─────────────────────────────────────────────────
        surface.blit(self.font_med.render('LEYENDA DE TERRENO', True, TEXT_WHITE), (x, y))
        y += 16

        terrain_items = [
            ('Bosque  (cobertura)', (32, 105,  38)),
            ('Rio     (obstaculo)', (42,  88, 138)),
            ('Objetivo (captura) ', (155, 22,  22)),
            ('Urbano  (cobertura)', (138, 130, 102)),
            ('Suministros        ', (185, 165,  22)),
        ]
        for t_name, t_col in terrain_items:
            iy = y + 5
            pygame.draw.rect(surface, t_col,      (x, iy - 4, 10, 10))
            pygame.draw.rect(surface, PANEL_BORDER, (x, iy - 4, 10, 10), 1)
            surface.blit(self.font_tiny.render(t_name, True, TEXT_DIM), (x + 14, y + 1))
            y += 14

        pygame.draw.line(surface, PANEL_BORDER, (x, y + 2), (x + bw, y + 2), 1)
        y += 10

        # ── Controls ──────────────────────────────────────────────────────
        surface.blit(self.font_med.render('CONTROLES', True, TEXT_WHITE), (x, y))
        y += 15
        hm_label = 'H        Mapa calor [ON]' if self.show_heatmap else 'H        Mapa calor'
        for ctrl in [
            'ESPACIO  Pausar / Reanudar',
            '+/-      Velocidad',
            'R        Reiniciar',
            'S        Guardar captura',
            hm_label,
            'ESC      Salir',
        ]:
            surface.blit(self.font_tiny.render(ctrl, True, TEXT_DIM), (x + 4, y))
            y += 13

        # ── Winner banner ──────────────────────────────────────────────────
        if self.winner:
            w_col  = {'Blue': (60,140,255), 'Red': RED_BRIGHT, 'Draw': GOLD}
            w_name = {'Blue': 'AZUL GANA!', 'Red': 'ROJO GANA!', 'Draw': 'EMPATE'}
            bc     = w_col.get(self.winner, TEXT_WHITE)
            bn     = w_name.get(self.winner, self.winner)
            by_    = self.win_h - 58

            banner_surf = pygame.Surface((256, 46), pygame.SRCALPHA)
            banner_surf.fill((bc[0]//5, bc[1]//5, bc[2]//5, 200))
            surface.blit(banner_surf, (x - 8, by_ - 4))
            pygame.draw.rect(surface, bc, (x - 8, by_ - 4, 256, 46), 2, border_radius=5)
            surface.blit(self.font_big.render(bn, True, bc), (x, by_ + 6))

    # ── Editor / setup mode ───────────────────────────────────────────────────

    def run_editor(self) -> tuple:
        """
        Pre-simulation setup screen.
        Returns (custom_units, weather) when the user presses ENTER to start.

        Controls
        --------
        1-6        Select arm  (1=Inf 2=Art 3=Cab 4=Com 5=Ing 6=Log)
        B / R      Select team (Blue / Red)
        O          Toggle obstacle-placement mode
        D          Toggle defensive state for placed units
        W          Cycle weather
        Left-click  Place unit / obstacle on map
        Right-click Remove unit / obstacle at cell
        ENTER / S  Start simulation
        ESC        Start with default units (no changes)
        """
        arm_order   = list(UnitType)
        sel_arm     = UnitType.INFANTRY
        sel_team    = 0
        place_obs   = False
        sel_def     = False
        cur_weather = self.sim.weather
        custom_units: list[Unit] = list(self.sim.units)  # start from default placement

        font_title = pygame.font.SysFont('Consolas', 17, bold=True)
        font_info  = pygame.font.SysFont('Consolas', 13)
        font_tiny2 = pygame.font.SysFont('Consolas', 11)

        # Working copy of terrain so we can preview obstacle edits
        edit_terrain = self.sim.terrain.copy()

        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return custom_units, cur_weather
                    elif event.key in (pygame.K_RETURN, pygame.K_s):
                        self.sim.weather  = cur_weather
                        self.sim.terrain  = edit_terrain
                        return custom_units, cur_weather
                    elif event.key == pygame.K_b:
                        sel_team = 0
                    elif event.key == pygame.K_r:
                        sel_team = 1
                    elif event.key == pygame.K_o:
                        place_obs = not place_obs
                    elif event.key == pygame.K_d:
                        sel_def = not sel_def
                    elif event.key == pygame.K_w:
                        cur_weather = Weather((int(cur_weather) + 1) % len(Weather))
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        sel_arm = UnitType.INFANTRY
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        sel_arm = UnitType.ARTILLERY
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        sel_arm = UnitType.CAVALRY
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        sel_arm = UnitType.COMMUNICATIONS
                    elif event.key in (pygame.K_5, pygame.K_KP5):
                        sel_arm = UnitType.ENGINEERING
                    elif event.key in (pygame.K_6, pygame.K_KP6):
                        sel_arm = UnitType.LOGISTICS
                    elif event.key == pygame.K_c:
                        custom_units.clear()
                        edit_terrain = self.sim.base_terrain.copy()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if mx >= self.map_w:
                        continue
                    col = mx // self.cell
                    row = my // self.cell
                    if not (0 <= row < self.rows and 0 <= col < self.cols):
                        continue

                    if event.button == 3:  # Right-click: remove
                        custom_units = [u for u in custom_units
                                        if not (u.row == row and u.col == col)]
                        if edit_terrain[row, col] == OBSTACLE:
                            edit_terrain[row, col] = EMPTY
                    elif event.button == 1:  # Left-click: place
                        if place_obs:
                            t = edit_terrain[row, col]
                            edit_terrain[row, col] = EMPTY if t == OBSTACLE else OBSTACLE
                        else:
                            if edit_terrain[row, col] == OBSTACLE:
                                continue
                            custom_units = [u for u in custom_units
                                            if not (u.row == row and u.col == col)]
                            from src.engine import UnitState as _US
                            new_u = Unit(team=sel_team, row=row, col=col,
                                         unit_type=sel_arm)
                            if sel_def:
                                new_u.state = _US.DEFENSIVE
                            custom_units.append(new_u)

            # ── Draw editor frame ────────────────────────────────────────
            self.screen.fill(BG_COLOR)

            # Terrain (with edits)
            for r in range(self.rows):
                for c2 in range(self.cols):
                    t    = int(edit_terrain[r, c2])
                    base = TERRAIN_BASE.get(t, TERRAIN_BASE[EMPTY])
                    pygame.draw.rect(self.screen, base,
                                     (c2 * self.cell, r * self.cell, self.cell, self.cell))

            # Grid lines
            if self.cell >= 7:
                gc = (35, 40, 50)
                for r in range(self.rows + 1):
                    pygame.draw.line(self.screen, gc,
                                     (0, r * self.cell), (self.map_w, r * self.cell))
                for c2 in range(self.cols + 1):
                    pygame.draw.line(self.screen, gc,
                                     (c2 * self.cell, 0), (c2 * self.cell, self.map_h))

            # Placed units
            for u in custom_units:
                cx, cy   = self._center(u.row, u.col)
                body_col = UNIT_BODY.get((u.team, u.unit_type), (128, 128, 128))
                pygame.draw.circle(self.screen, body_col, (cx, cy), max(3, self.cell // 2 - 1))
                lbl = font_tiny2.render(ARM_LABELS[u.unit_type], True, TEXT_WHITE)
                self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

            # Mouse hover highlight
            mx, my = pygame.mouse.get_pos()
            if mx < self.map_w:
                hc = (mx // self.cell) * self.cell
                hr = (my // self.cell) * self.cell
                hl = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
                hcol = (255, 80, 80, 80) if place_obs else (
                    (60, 140, 255, 80) if sel_team == 0 else (255, 60, 60, 80))
                hl.fill(hcol)
                self.screen.blit(hl, (hc, hr))

            # ── Side panel ────────────────────────────────────────────────
            px = self.map_w
            pygame.draw.rect(self.screen, PANEL_BG, (px, 0, self.PANEL_WIDTH, self.win_h))
            pygame.draw.line(self.screen, PANEL_BORDER, (px, 0), (px, self.win_h), 2)

            ex, ey = px + 12, 14
            self.screen.blit(font_title.render('MODO EDITOR', True, GOLD), (ex, ey))
            ey += 26

            pygame.draw.line(self.screen, PANEL_BORDER, (ex, ey), (ex + 270, ey), 1)
            ey += 8

            def erow(txt, col=TEXT_DIM):
                nonlocal ey
                self.screen.blit(font_info.render(txt, True, col), (ex, ey))
                ey += 16

            # Team
            t_col = (60, 140, 255) if sel_team == 0 else RED_BRIGHT
            t_lbl = 'AZUL' if sel_team == 0 else 'ROJO'
            erow(f'Equipo : {t_lbl}  (B/R)', t_col)

            # Arm
            arm_col2 = UNIT_BODY.get((sel_team, sel_arm), TEXT_WHITE)
            erow(f'Arma   : {ARM_FULL[sel_arm]}', arm_col2)

            # Mode
            mode_lbl = 'OBSTACULO' if place_obs else 'UNIDAD'
            mode_col = (255, 200, 0) if place_obs else TEXT_WHITE
            erow(f'Modo   : {mode_lbl}  (O)', mode_col)

            # Defensive
            def_lbl = 'SI' if sel_def else 'NO'
            erow(f'Defensivo: {def_lbl}  (D)', (0, 200, 255) if sel_def else TEXT_DIM)

            # Weather
            wx_col2 = WEATHER_COLOR.get(cur_weather, TEXT_WHITE)
            erow(f'Clima  : {WEATHER_NAMES[cur_weather]}  (W)', wx_col2)

            ey += 4
            pygame.draw.line(self.screen, PANEL_BORDER, (ex, ey), (ex + 270, ey), 1)
            ey += 8

            # Arm selector
            self.screen.blit(font_info.render('ARMAS  (teclas 1-6)', True, TEXT_WHITE), (ex, ey))
            ey += 16
            for i, ut in enumerate(arm_order):
                col3 = UNIT_BODY.get((sel_team, ut), TEXT_DIM)
                marker = '>' if ut == sel_arm else ' '
                self.screen.blit(font_tiny2.render(
                    f'{marker} {i+1}. {ARM_FULL[ut]}', True,
                    col3 if ut == sel_arm else TEXT_DIM), (ex + 4, ey))
                ey += 14

            ey += 4
            pygame.draw.line(self.screen, PANEL_BORDER, (ex, ey), (ex + 270, ey), 1)
            ey += 8

            # Unit count summary
            blue_n = sum(1 for u in custom_units if u.team == 0)
            red_n  = sum(1 for u in custom_units if u.team == 1)
            erow(f'Azul: {blue_n} unidades', (60, 140, 255))
            erow(f'Rojo: {red_n} unidades',  RED_BRIGHT)

            ey += 4
            pygame.draw.line(self.screen, PANEL_BORDER, (ex, ey), (ex + 270, ey), 1)
            ey += 8

            # Controls help
            for ctrl in [
                'Click izq  Colocar',
                'Click der  Borrar',
                'C          Limpiar todo',
                'ENTER/S    Iniciar sim',
                'ESC        Usar default',
            ]:
                self.screen.blit(font_tiny2.render(ctrl, True, TEXT_DIM), (ex + 2, ey))
                ey += 13

            pygame.display.flip()

        return custom_units, cur_weather

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 60) -> CASimulator:
        """Run the interactive simulation window. Returns simulator on exit."""
        stats = self.sim._collect_stats()

        while self.running:
            dt = self.clock.tick(max(self.fps, 1)) / 1000.0

            # Animation timing
            self.anim_timer += dt
            if self.anim_timer >= self.ANIM_INTERVAL:
                self.anim_frame = 1 - self.anim_frame
                self.anim_timer = 0.0

            # ── Events ────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.fps = min(60, self.fps + 2)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fps = max(1, self.fps - 2)
                    elif event.key == pygame.K_r:
                        self.sim = CASimulator(self.terrain, params=self.sim.params)
                        self.step  = 0
                        self.winner = None
                        self.particles.clear()
                        self.tracers.clear()
                        self.heat_map[:] = 0
                        self.blue_history.clear()
                        self.red_history.clear()
                        stats = self.sim._collect_stats()
                    elif event.key == pygame.K_s:
                        pygame.image.save(
                            self.screen,
                            f'outputs/screenshot_{self.step:03d}.png'
                        )
                    elif event.key == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap

            # ── Simulation tick ────────────────────────────────────────────
            if not self.paused and self.winner is None and self.step < max_steps:
                _, _, stats = self.sim.step()
                self._process_combat_events(self.sim.combat_events)
                self.step  += 1
                self.winner = self.sim.get_winner()

                # Track battle history for chart
                self.blue_history.append(stats['blue_alive'])
                self.red_history.append(stats['red_alive'])
                if self.step == 1:
                    self.max_units = max(stats['blue_alive'],
                                        stats['red_alive'], 1)

            # ── Render ────────────────────────────────────────────────────
            self.screen.fill(BG_COLOR)

            # Terrain
            self.screen.blit(self.terrain_surf, (0, 0))

            # Subtle grid lines (only when cells are big enough)
            if self.cell >= 8:
                grid_col = (28, 32, 40)
                for r in range(self.rows + 1):
                    pygame.draw.line(self.screen, grid_col,
                                     (0, r * self.cell), (self.map_w, r * self.cell))
                for c in range(self.cols + 1):
                    pygame.draw.line(self.screen, grid_col,
                                     (c * self.cell, 0), (c * self.cell, self.map_h))

            # Fog of war (skip when heat map is active for readability)
            if not self.show_heatmap:
                self._draw_fog(self.screen)

            # Heat map overlay (toggled with H)
            if self.show_heatmap:
                self._draw_heatmap(self.screen)

            # Units (pixel-art sprites)
            self._draw_units(self.screen)

            # Tracers + particles
            self._draw_fx(self.screen, dt)

            # Stats panel
            self._draw_panel(self.screen, stats)

            pygame.display.flip()

        pygame.quit()
        return self.sim
