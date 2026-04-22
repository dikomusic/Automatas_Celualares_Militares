import os
import sys
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mapper    import MapLoader
from src.engine    import CASimulator
from src.optimizer import RechenbergOptimizer

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Terrain colour palette (RGB, 0-1) -----------------------------------------
TERRAIN_RGB = {
    0: [0.08, 0.08, 0.10],   # Empty    - near-black
    1: [0.13, 0.45, 0.13],   # Forest   - dark green
    2: [0.25, 0.45, 0.65],   # Obstacle - steel blue (river)
    3: [0.75, 0.10, 0.10],   # Objective- crimson
    4: [0.65, 0.62, 0.48],   # Urban    - tan (contour lines)
    5: [0.90, 0.80, 0.10],   # Supply   - gold
}


# -- Frame renderer ------------------------------------------------------------
def render_frame(terrain, blue_grid, red_grid, stats, step, save_path):
    fig = plt.figure(figsize=(14, 7), facecolor='#0d1117')
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.03)
    ax_map   = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    # Build RGB terrain image
    rows, cols = terrain.shape
    rgb = np.zeros((rows, cols, 3))
    for t, color in TERRAIN_RGB.items():
        mask = terrain == t
        rgb[mask] = color

    ax_map.imshow(rgb, interpolation='nearest')
    ax_map.set_facecolor('#0d1117')

    # Blue units (attacker) - cyan squares
    br, bc = np.where(blue_grid > 0)
    if len(br):
        ax_map.scatter(bc, br, c='cyan', s=18, marker='s',
                       alpha=0.95, linewidths=0, zorder=4)

    # Red units (defender) - red triangles
    rr, rc = np.where(red_grid > 0)
    if len(rr):
        ax_map.scatter(rc, rr, c='#ff4444', s=18, marker='^',
                       alpha=0.95, linewidths=0, zorder=4)

    ax_map.set_title(f'Tactical Simulation  -  Step {step:03d}',
                     color='white', fontsize=13, fontweight='bold', pad=8)
    ax_map.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='cyan',    label=f"Blue (Attacker) : {stats['blue_alive']} units"),
        mpatches.Patch(facecolor='#ff4444', label=f"Red  (Defender) : {stats['red_alive']} units"),
        mpatches.Patch(facecolor='#217021', label='Forest  (cover)'),
        mpatches.Patch(facecolor='#4072A6', label='River   (obstacle)'),
        mpatches.Patch(facecolor='#BF1A1A', label='Objective'),
        mpatches.Patch(facecolor='#E6CC00', label='Supply depot'),
    ]
    ax_map.legend(handles=legend_elements, loc='lower left',
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white', fontsize=8, framealpha=0.9)

    # -- Stats panel -----------------------------------------------------------
    ax_stats.set_facecolor('#161b22')
    ax_stats.axis('off')

    def bar(ax, y, val, color, label, max_val=100):
        ax.barh(y, val / max_val, color=color, height=0.06,
                left=0.05, alpha=0.85)
        ax.text(0.05 + val / max_val + 0.02, y, f'{val:.0f}',
                color='white', va='center', fontsize=7)
        ax.text(0.04, y, label, color='#aaaaaa', va='center',
                ha='right', fontsize=7)

    lines = [
        ('', ''),
        ('BLUE TEAM', ''),
        ('  Units   ', f"{stats['blue_alive']}"),
        ('  HP      ', f"{stats['blue_avg_health']:.1f}"),
        ('  Ammo    ', f"{stats['blue_avg_ammo']:.1f}"),
        ('  Morale  ', f"{stats['blue_avg_morale']:.2f}"),
        ('', ''),
        ('RED TEAM', ''),
        ('  Units   ', f"{stats['red_alive']}"),
        ('  HP      ', f"{stats['red_avg_health']:.1f}"),
        ('  Ammo    ', f"{stats['red_avg_ammo']:.1f}"),
        ('  Morale  ', f"{stats['red_avg_morale']:.2f}"),
    ]
    text = '\n'.join(
        f"{k}{v}" if k.strip() else ''
        for k, v in lines
    )
    ax_stats.text(0.08, 0.92, f'STEP {step:03d}',
                  transform=ax_stats.transAxes, color='#58a6ff',
                  fontsize=14, fontweight='bold', va='top')
    ax_stats.text(0.08, 0.82, text,
                  transform=ax_stats.transAxes, color='white',
                  fontsize=9, va='top', fontfamily='monospace',
                  linespacing=1.6)

    plt.savefig(save_path, bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)


# -- Analytical report ---------------------------------------------------------
def generate_report(sim: CASimulator, opt_results: list,
                    best_params: dict, path: str):
    units = sim.units
    blue_all  = [u for u in units if u.team == 0]
    red_all   = [u for u in units if u.team == 1]
    blue_alive = [u for u in blue_all if u.alive]
    red_alive  = [u for u in red_all  if u.alive]

    winner     = sim.get_winner() or 'Draw'
    wins       = sum(1 for r in opt_results if r['won'])
    total_iter = len(opt_results)
    success_rt = wins / total_iter if total_iter else 0.0

    sep = '=' * 64

    lines = [
        sep,
        '  TACTICAL SIMULATION - ANALYTICAL REPORT',
        sep,
        '',
        f'  MISSION OUTCOME : {winner.upper()} {"VICTORY" if winner != "Draw" else ""}',
        '',
        '  EVOLUTIONARY OPTIMIZATION  (Rechenberg 1/5 Rule)',
        f'    Iterations executed  : {total_iter}',
        f'    Successful runs      : {wins}',
        f'    Success rate         : {success_rt:.1%}',
        '',
        '  OPTIMIZED TACTICAL PARAMETERS',
        f'    Aggressiveness       : {best_params.get("aggressiveness", 0):.3f}',
        f'    Cover Seeking        : {best_params.get("cover_seeking", 0):.3f}',
        f'    Teamwork             : {best_params.get("teamwork", 0):.3f}',
        '',
        '  CASUALTY STATISTICS',
        '    Blue (Attacker)',
        f'      Initial units      : {len(blue_all)}',
        f'      Surviving units    : {len(blue_alive)}',
        f'      Casualties         : {len(blue_all) - len(blue_alive)}'
        f'  ({(len(blue_all)-len(blue_alive))/max(len(blue_all),1)*100:.1f}%)',
        '    Red (Defender)',
        f'      Initial units      : {len(red_all)}',
        f'      Surviving units    : {len(red_alive)}',
        f'      Casualties         : {len(red_all) - len(red_alive)}'
        f'  ({(len(red_all)-len(red_alive))/max(len(red_all),1)*100:.1f}%)',
        '',
        '  RESOURCE CONSUMPTION (final state)',
        f'    Avg ammo  Blue       : {np.mean([u.ammo for u in blue_all]):.1f} / 30',
        f'    Avg ammo  Red        : {np.mean([u.ammo for u in red_all]):.1f} / 30',
        f'    Avg morale Blue      : {np.mean([u.morale for u in blue_alive] or [0]):.2f}',
        f'    Avg morale Red       : {np.mean([u.morale for u in red_alive]  or [0]):.2f}',
        '',
        '  ITERATION LOG',
    ]

    for r in opt_results:
        p   = r['params']
        tag = 'WIN ' if r['won'] else 'LOSS'
        lines.append(
            f"    Iter {r['iteration']:2d}: {tag} | "
            f"Aggr={p['aggressiveness']:.2f}  "
            f"Cover={p['cover_seeking']:.2f}  "
            f"Team={p['teamwork']:.2f}  |  "
            f"Blue casualties: {r['casualties']}"
        )

    lines += ['', sep]
    report = '\n'.join(lines)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)

    print('\n' + report)


# -- Main entry point ----------------------------------------------------------
def run_simulation(steps: int = 30, n_opt_iter: int = 10):
    print('=' * 60)
    print('  Military Tactical Simulator - Open House 2026')
    print('=' * 60)

    loader  = MapLoader('assets/image.png', grid_width=80)
    terrain = loader.get_grid()
    print(f'\n  Grid size : {terrain.shape[0]} rows x {terrain.shape[1]} cols')

    unique, counts = np.unique(terrain, return_counts=True)
    for u, c in zip(unique, counts):
        names = {0:'Empty',1:'Forest',2:'Obstacle',3:'Objective',4:'Urban',5:'Supply'}
        print(f'    {names.get(u, u)}: {c} cells ({c/terrain.size*100:.1f}%)')

    # Phase 3: Rechenberg optimization
    optimizer = RechenbergOptimizer(terrain, n_iterations=n_opt_iter, sim_steps=steps)
    best_params, _, opt_results = optimizer.optimize()

    # Phase 4: Final simulation with best params
    print(f'\n--- Final simulation with optimized parameters ({steps} steps) ---')
    sim = CASimulator(terrain, params=best_params)

    last_frame_path = None
    final_step      = steps - 1

    for i in range(steps):
        blue_grid, red_grid, stats = sim.step()

        save_path = os.path.join(OUTPUT_DIR, f'frame_{i:03d}.png')
        render_frame(terrain, blue_grid, red_grid, stats, i, save_path)
        last_frame_path = save_path

        if i % 5 == 0 or i == steps - 1:
            print(f'  Step {i:3d} | Blue: {stats["blue_alive"]:3d} | '
                  f'Red: {stats["red_alive"]:3d} | '
                  f'Blue HP: {stats["blue_avg_health"]:5.1f} | '
                  f'Red HP: {stats["red_avg_health"]:5.1f}')

        winner = sim.get_winner()
        if winner is not None:
            print(f'\n  *** Mission ended at step {i}: {winner} wins! ***')
            final_step = i
            # Duplicate last frame for remaining slots
            for j in range(i + 1, steps):
                shutil.copy(last_frame_path,
                            os.path.join(OUTPUT_DIR, f'frame_{j:03d}.png'))
            break

    # Phase 4: Analytical report
    report_path = os.path.join(OUTPUT_DIR, 'tactical_report.txt')
    generate_report(sim, opt_results, best_params, report_path)

    print(f'\n  Frames  -> {OUTPUT_DIR}/ (frame_000 ... frame_{final_step:03d})')
    print(f'  Report  -> {report_path}')
    print('\n  Simulation complete.')


if __name__ == '__main__':
    run_simulation(steps=30, n_opt_iter=10)
