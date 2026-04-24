import sys
import os
import copy
import pygame
from typing import List, Optional

from src.environment.map_parser import MapParser
from src.environment.grid import TacticalGrid
from src.core.engine import SimulationEngine
from src.entities.unit import Unit
from src.ui.renderer import TacticalRenderer
from src.ui.analytics import AnalyticsManager

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'maps_dir': 'data/maps',
    'max_turns': 2000,
    'cell_size': 10,
    'sim_speed': 30
}

class TacticalApp:
    """
    Main Application Controller for the Military Tactical Simulator.
    
    Orchestrates the lifecycle of the simulation, managing user input, 
    engine updates, and rendering.
    """

    def __init__(self):
        # 1. System Components
        self.analytics = AnalyticsManager()
        self.parser = MapParser()
        self.clock = pygame.time.Clock()
        
        # 2. Map Management
        self.map_files = self._get_available_maps()
        if not self.map_files:
            raise RuntimeError("No map files found in data/maps/")
        
        self.map_idx = 0
        self.current_map_path = ""
        self.grid: Optional[TacticalGrid] = None
        
        # 3. Simulation State
        self.is_running = True
        self.is_paused = True
        self.active_brush = 'blue_team'  # blue_team, red_team, eraser
        self.turn = 0
        self.combat_traces = []
        
        # 4. Engine & Renderer Initialization
        self._load_map(self.map_idx)
        self.renderer = TacticalRenderer(self.grid.width, self.grid.height, CONFIG['cell_size'])
        self.engine = SimulationEngine(self.grid, [], [], self._get_default_depots())

    def _get_available_maps(self) -> List[str]:
        """Returns list of .png files in the maps directory."""
        if not os.path.exists(CONFIG['maps_dir']):
            return []
        return sorted([f for f in os.listdir(CONFIG['maps_dir']) if f.endswith('.png')])

    def _get_default_depots(self) -> List[tuple]:
        """Calculates center-map supply points."""
        return [(self.grid.width // 2, self.grid.height // 2)]

    def _load_map(self, idx: int):
        """Loads a specific map from the directory."""
        self.map_idx = idx % len(self.map_files)
        self.current_map_path = os.path.join(CONFIG['maps_dir'], self.map_files[self.map_idx])
        self.grid = self.parser.parse_image_to_grid(self.current_map_path)
        
        # Update existing engine if it exists
        if hasattr(self, 'engine'):
            self.engine.grid = self.grid
            # Purge out-of-bounds units
            self.engine.units = [u for u in self.engine.units if self.grid.is_in_bounds(u.x, u.y)]
        
        print(f"[*] Map loaded: {self.map_files[self.map_idx]}")

    def _handle_events(self):
        """Dispatches input events to specialized handlers."""
        for event in self.renderer.handle_events():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                self._on_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mousedown(event)

    def _on_keydown(self, event):
        """Handles keypress-based commands."""
        if event.key == pygame.K_SPACE:
            self.is_paused = not self.is_paused
        elif event.key == pygame.K_RIGHT and self.is_paused:
            self._step_simulation()
        elif event.key == pygame.K_1:
            self.active_brush = 'blue_team'
        elif event.key == pygame.K_2:
            self.active_brush = 'red_team'
        elif event.key == pygame.K_3:
            self.active_brush = 'eraser'
        elif event.key == pygame.K_m:
            self._load_map(self.map_idx + 1)
        elif event.key == pygame.K_c:
            self.engine.units = []
            self.turn = 0
            print("[*] Tactical environment cleared.")

    def _on_mousedown(self, event):
        """Handles manual troop placement and removal, or button/map interaction."""
        # 1. Check for UI Buttons
        btn_key = self.renderer.get_button_clicked(event.pos)
        if btn_key == 'play':
            self.is_paused = not self.is_paused
            return
        elif btn_key == 'step':
            if self.is_paused:
                self._step_simulation()
            return

        # 2. Check for Map Selection (Sidebar)
        map_idx = self.renderer.get_map_selector_index(event.pos)
        if map_idx is not None and map_idx < len(self.map_files):
            self._load_map(map_idx)
            return
        gx, gy = self.renderer.get_grid_coords(event.pos)
        if not self.grid.is_in_bounds(gx, gy):
            return

        if self.active_brush == 'eraser':
            self.engine.units = [u for u in self.engine.units if (u.x, u.y) != (gx, gy)]
        else:
            # Check for existing unit
            if not any(u.x == gx and u.y == gy for u in self.engine.units):
                new_unit = Unit(faction=self.active_brush, x=gx, y=gy)
                # Apply faction doctrines
                if self.active_brush == 'blue_team':
                    new_unit.aggressiveness, new_unit.teamwork = 0.8, 0.4
                else:
                    new_unit.aggressiveness, new_unit.teamwork = 0.2, 0.8
                self.engine.units.append(new_unit)

    def _step_simulation(self):
        """Executes one simulation tick and logs events."""
        tick_data = self.engine.tick()
        self.combat_traces.extend(tick_data['shots'])
        
        # Log to analytics for post-mission debrief
        for unit in self.engine.units:
            self.analytics.log_event(self.turn, 'move', unit.faction, (unit.x, unit.y))
        for shot in tick_data['shots']:
            self.analytics.log_event(self.turn, 'shot', shot['faction'], shot['attacker'])
        for death in tick_data['deaths']:
            self.analytics.log_event(self.turn, 'death', death['faction'], (death['x'], death['y']))
        
        self.turn += 1
        if self.turn % 100 == 0:
            print(f"[*] Turn {self.turn} Progress Logged.")

    def run(self):
        """Entry point for the simulation loop."""
        print("[*] Tactical Command Center Online.")
        print("[*] Controls: SPACE(Pause), RIGHT(Step), 1/2/3(Brushes), M(Map), C(Clear)")

        while self.is_running:
            self._handle_events()
            
            self.combat_traces = [] # Clear traces every frame for rendering
            if not self.is_paused:
                self._step_simulation()
                if self.turn >= CONFIG['max_turns']:
                    self.is_paused = True

            # Render Phase
            self.renderer.render_frame(
                grid=self.grid,
                units=self.engine.units,
                combat_events=self.combat_traces,
                turn=self.turn,
                is_paused=self.is_paused,
                brush=self.active_brush,
                map_name=self.map_files[self.map_idx],
                available_maps=self.map_files
            )

            self.clock.tick(CONFIG['sim_speed'])

        self._finalize()

    def _finalize(self):
        """Exports data and closes systems."""
        if self.turn > 0:
            print("\n[*] Exporting Mission Analytics...")
            self.analytics.generate_mission_report(self.engine.units, self.engine.units, self.turn)
            self.analytics.generate_heatmaps(self.current_map_path, (self.grid.width, self.grid.height))
        
        self.renderer.close()
        print("[+] All systems offline. Shutdown complete.")

if __name__ == "__main__":
    try:
        app = TacticalApp()
        app.run()
    except Exception as e:
        print(f"[!] Critical Error during deployment: {e}")
        sys.exit(1)
