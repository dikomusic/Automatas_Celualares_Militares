import pygame
from typing import List, Tuple, Dict, Optional
from src.environment.grid import TacticalGrid, TerrainType
from src.entities.unit import Unit

from src.ui.elements import Button

class TacticalRenderer:
    """
    Commander's Tactical Visualizer - Interactive Version.
    
    Renders a Left Sidebar for statistics and controls, and the Tactical Grid 
    to the right. Supports manual placement and time-control feedback.
    """

    SIDEBAR_WIDTH = 250

    COLORS = {
        TerrainType.FOREST: (34, 139, 34),
        TerrainType.RIVER:  (0, 105, 148),
        TerrainType.URBAN:  (105, 105, 105),
        TerrainType.PLAIN:  (210, 180, 140),
        'blue_team':        (0, 120, 255),
        'red_team':         (255, 60, 60),
        'SHOT_TRACE':       (255, 255, 0),
        'PINNED':           (255, 255, 255),
        'HUD_BG':           (15, 15, 20),
        'SIDEBAR_BG':       (30, 30, 35),
        'TEXT':             (230, 230, 230),
        'HIGHLIGHT':        (0, 255, 0),
        'BTN_PLAY':         (34, 139, 34),
        'BTN_STEP':         (70, 70, 180)
    }

    def __init__(self, width: int, height: int, cell_size: int = 10):
        pygame.init()
        self.cell_size = cell_size
        self.grid_w = width
        self.grid_h = height
        
        self.screen_width = (width * cell_size) + self.SIDEBAR_WIDTH
        self.screen_height = height * cell_size
        
        # Ensure minimum height for sidebar content
        if self.screen_height < 600:
            self.screen_height = 600
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Military Tactical Simulator - Command Center")
        
        self.font_main = pygame.font.SysFont("monospace", 14)
        self.font_bold = pygame.font.SysFont("monospace", 16, bold=True)
        self.clock = pygame.time.Clock()

        # Initialize Buttons
        self.buttons = {
            'play': Button(20, 120, 100, 30, "PLAY/PAUSE", self.COLORS['BTN_PLAY']),
            'step': Button(130, 120, 90, 30, "STEP >", self.COLORS['BTN_STEP'])
        }

    def render_frame(self, grid: TacticalGrid, units: List[Unit], combat_events: List[Dict], 
                     turn: int, is_paused: bool, brush: str, map_name: str, available_maps: List[str]):
        """Draws the sidebar and the tactical grid."""
        self.screen.fill(self.COLORS['SIDEBAR_BG'])

        # 1. Draw Tactical Grid (Right Side)
        grid_surface = pygame.Surface((grid.width * self.cell_size, grid.height * self.cell_size))
        for y in range(grid.height):
            for x in range(grid.width):
                terrain = grid.matrix[y, x]
                color = self.COLORS.get(terrain, self.COLORS[TerrainType.PLAIN])
                pygame.draw.rect(grid_surface, color, 
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        # 2. Draw Combat Tracers
        for shot in combat_events:
            ax, ay = shot['attacker']
            dx, dy = shot['defender']
            start = (ax * self.cell_size + self.cell_size // 2, ay * self.cell_size + self.cell_size // 2)
            end   = (dx * self.cell_size + self.cell_size // 2, dy * self.cell_size + self.cell_size // 2)
            pygame.draw.line(grid_surface, self.COLORS['SHOT_TRACE'], start, end, 2)

        # 3. Draw Units
        for unit in units:
            if not unit.is_alive: continue
            center = (unit.x * self.cell_size + self.cell_size // 2,
                      unit.y * self.cell_size + self.cell_size // 2)
            color = self.COLORS.get(unit.faction, (128, 128, 128))
            pygame.draw.circle(grid_surface, color, center, self.cell_size // 2 - 1)
            if unit.pinned_timer > 0:
                pygame.draw.circle(grid_surface, self.COLORS['PINNED'], center, self.cell_size // 3, 1)

        self.screen.blit(grid_surface, (self.SIDEBAR_WIDTH, 0))

        # 4. Draw Sidebar (Left Side)
        self._draw_sidebar(units, turn, is_paused, brush, map_name, available_maps)

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_sidebar(self, units: List[Unit], turn: int, is_paused: bool, brush: str, 
                      current_map: str, available_maps: List[str]):
        y_off = 20
        margin = 20
        
        def write(text, color=self.COLORS['TEXT'], bold=False):
            nonlocal y_off
            f = self.font_bold if bold else self.font_main
            surf = f.render(text, True, color)
            self.screen.blit(surf, (margin, y_off))
            y_off += 22

        write("TACTICAL COMMAND", self.COLORS['HIGHLIGHT'], bold=True)
        y_off += 10
        
        state_str = "[PAUSED]" if is_paused else "[RUNNING]"
        state_col = (255, 100, 100) if is_paused else (100, 255, 100)
        write(f"STATUS: {state_str}", state_col, bold=True)
        write(f"TURN:   {turn:04}")
        
        # Draw Buttons
        y_off = 160 # Push content below buttons
        for btn in self.buttons.values():
            btn.draw(self.screen)

        y_off += 5
        write("MAP SELECTOR:", bold=True)
        for i, m_name in enumerate(available_maps):
            color = self.COLORS['HIGHLIGHT'] if m_name == current_map else self.COLORS['TEXT']
            indicator = "> " if m_name == current_map else "  "
            write(f"{indicator}{m_name[:18]}", color)
        
        y_off += 15
        write("ACTIVE BRUSH:", bold=True)
        brush_col = self.COLORS.get(brush, self.COLORS['TEXT'])
        write(f"> {brush.upper()}", brush_col)
        
        y_off += 25
        # Stats Calculation
        blue = [u for u in units if u.faction == 'blue_team' and u.is_alive]
        red = [u for u in units if u.faction == 'red_team' and u.is_alive]
        
        write("BLUE TEAM STATS", self.COLORS['blue_team'], bold=True)
        write(f" Units:  {len(blue)}")
        write(f" Avg HP: {sum(u.hp for u in blue)/max(1, len(blue)):.1f}")
        write(f" Morale: {sum(u.morale for u in blue)/max(1, len(blue)):.2f}")
        
        y_off += 15
        write("RED TEAM STATS", self.COLORS['red_team'], bold=True)
        write(f" Units:  {len(red)}")
        write(f" Avg HP: {sum(u.hp for u in red)/max(1, len(red)):.1f}")
        write(f" Morale: {sum(u.morale for u in red)/max(1, len(red)):.2f}")
        
        # Controls Hint at bottom
        y_off = self.screen_height - 180
        write("CONTROLS:", bold=True)
        write("SPACE : Play/Pause")
        write("RIGHT : Step Forward")
        write("1/2/3 : Blue/Red/Eraser")
        write("M     : Cycle Maps")
        write("C     : Clear All Units")
        write("L-CLK : Place/Remove/Buttons")

    def get_button_clicked(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Returns the key of the button clicked, if any."""
        for key, btn in self.buttons.items():
            if btn.is_clicked(mouse_pos):
                return key
        return None

    def get_map_selector_index(self, mouse_pos: Tuple[int, int]) -> Optional[int]:
        """Returns the index of the map clicked in the sidebar, if any."""
        mx, my = mouse_pos
        if mx > self.SIDEBAR_WIDTH:
            return None
        
        # Updated start_y based on new layout (with buttons)
        # y=20 + header(22) + 10 + status(22) + turn(22) + buttons_area(~50) + 5 + "MAP SELECTOR"(22) = 187
        start_y = 187
        if my < start_y:
            return None
        
        idx = (my - start_y) // 22
        return idx

    def get_grid_coords(self, mouse_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Translates screen mouse position to grid (x, y)."""
        mx, my = mouse_pos
        gx = (mx - self.SIDEBAR_WIDTH) // self.cell_size
        gy = my // self.cell_size
        return gx, gy

    def handle_events(self) -> List[pygame.event.Event]:
        return pygame.event.get()

    def close(self):
        pygame.quit()
