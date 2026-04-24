# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the simulation

```bash
# Interactive Pygame window (default)
python main.py

# Export PNG frames to outputs/
python main.py --mode export

# Custom steps and optimizer iterations
python main.py --steps 60 --opt-iter 10 --mode live
```

Dependencies: `opencv-python`, `numpy`, `matplotlib`, `pygame`, `scipy`.  
Python 3.12 on Windows — avoid unicode box-drawing characters in print statements (cp1252 terminal encoding will error).

## Architecture

The pipeline runs in a fixed order: **mapper → optimizer → simulator → visualizer**.

```
main.py
  └─ MapLoader (src/mapper.py)       image → NumPy terrain grid
  └─ RechenbergOptimizer (src/optimizer.py)  finds best tactical params
  └─ CASimulator (src/engine.py)     runs the cellular automaton
  └─ PygameVisualizer (src/visualizer.py)    real-time rendering
```

### src/mapper.py — Terrain digitisation
Reads `assets/map_test.png`, resizes to `grid_width=80` (produces a 64×80 grid), converts to HSV, and segments into 6 terrain states (0–5). Mask application order matters — later masks overwrite earlier ones. Terrain constants are defined here AND mirrored at the top of `engine.py`; keep them in sync.

### src/engine.py — Cellular automaton core
- **Terrain constants** (EMPTY=0 … SUPPLY=5), `MOVE_COST`, and `COVER_BONUS` dicts are module-level and imported by `visualizer.py`.
- **`dijkstra(terrain, seeds)`** precomputes weighted distance maps used for autonomous navigation. Called once per simulator instance.
- **`CASimulator`** holds a flat `list[Unit]`. Each `step()` does: shuffle-move → combat → supply → morale decay → collect stats. Returns `(blue_grid, red_grid, stats)` and stores `combat_events` (consumed by the visualizer for tracers/particles).
- **Unit types**: Infantry / Sniper / Tank — stats (range, damage, HP, speed) are in the `UNIT_STATS` dict keyed by `UnitType`. Snipers use a larger vision/fire range; Tanks move at 0.6× speed and take 0.6× damage.
- **Movement scoring**: `score = -aggressiveness×BFS_dist + cover_seeking×cover×12 + teamwork_cohesion`. Units with HP<25%, ammo≤2, or morale<0.15 switch to retreat mode (flee from nearest visible enemy toward cover).
- **Combat uses Chebyshev distance** (`max(|Δrow|, |Δcol|)`) not Manhattan, so units engage within a square area regardless of column spread on the wide map.
- **Victory**: Blue wins when any unit occupies an OBJECTIVE cell with `row < rows//2` (top-half only, to exclude stray objective pixels near the bottom of the image).

### src/optimizer.py — Rechenberg 1/5 rule
Perturbs three params (`aggressiveness`, `cover_seeking`, `teamwork`) with Gaussian noise σ, runs `n_iterations` independent simulations, counts Blue wins. If success rate > 20%: `σ × 1.22`; else `σ / 1.22`. Returns the best-performing param set to pass into the final `CASimulator`.

### src/visualizer.py — Pygame real-time display
Pre-renders the terrain surface once (`_build_terrain_surface`). Each frame: blit terrain → fog-of-war overlay (semi-transparent black cleared around Blue vision radii) → draw units → draw tracers/particles (from `sim.combat_events`) → stats panel. Unit shapes: Infantry=square, Sniper=diamond, Tank=hexagon. Keyboard: SPACE=pause, +/-=FPS, R=restart, S=screenshot, ESC=quit.

## Key constants to know

| Constant | Location | Value | Notes |
|---|---|---|---|
| `grid_width` | `main.py` | 80 | Controls grid size (64×80). Changing this shifts spawn rows and objective positions. |
| Blue spawn | `engine.py _place_units` | `rows × 0.52` | ~row 33 on 64-row grid |
| Red spawn | `engine.py _place_units` | `rows × 0.22` | ~row 14, near top objectives |
| `FIRE_RANGE` | `engine.py` per unit type | 12 / 22 / 8 | Chebyshev distance |
| Victory row limit | `engine.py get_winner` | `rows // 2` | Only top-half OBJECTIVE cells count |

## Outputs

- `outputs/frame_NNN.png` — PNG frames (export mode only)
- `outputs/reporte_tactico.txt` — UTF-8 analytical report (always generated)
- `outputs/screenshot_NNN.png` — manual screenshots from Pygame (S key)
