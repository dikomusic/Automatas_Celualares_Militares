import os
import sys
import shutil
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mapper      import MapLoader
from src.engine      import CASimulator, UnitType, Weather, WEATHER_NAMES
from src.optimizer    import RechenbergOptimizer
from src.visualizer   import PygameVisualizer

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


# -- Frame renderer (for --export mode) ----------------------------------------
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

    ax_map.set_title(f'Simulación Táctica  -  Paso {step:03d}',
                     color='white', fontsize=13, fontweight='bold', pad=8)
    ax_map.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='cyan',    label=f"Azul (Atacante) : {stats['blue_alive']} unidades"),
        mpatches.Patch(facecolor='#ff4444', label=f"Rojo (Defensor) : {stats['red_alive']} unidades"),
        mpatches.Patch(facecolor='#217021', label='Bosque  (cobertura)'),
        mpatches.Patch(facecolor='#4072A6', label='Río    (obstáculo)'),
        mpatches.Patch(facecolor='#BF1A1A', label='Objetivo'),
        mpatches.Patch(facecolor='#E6CC00', label='Depósito de suministros'),
    ]
    ax_map.legend(handles=legend_elements, loc='lower left',
                  facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white', fontsize=8, framealpha=0.9)

    # -- Stats panel -----------------------------------------------------------
    ax_stats.set_facecolor('#161b22')
    ax_stats.axis('off')

    lines = [
        ('', ''),
        ('EQUIPO AZUL', ''),
        ('  Unidades', f"{stats['blue_alive']}"),
        ('  Salud   ', f"{stats['blue_avg_health']:.1f}"),
        ('  Munición', f"{stats['blue_avg_ammo']:.1f}"),
        ('  Moral   ', f"{stats['blue_avg_morale']:.2f}"),
        ('', ''),
        ('EQUIPO ROJO', ''),
        ('  Unidades', f"{stats['red_alive']}"),
        ('  Salud   ', f"{stats['red_avg_health']:.1f}"),
        ('  Munición', f"{stats['red_avg_ammo']:.1f}"),
        ('  Moral   ', f"{stats['red_avg_morale']:.2f}"),
    ]
    text = '\n'.join(
        f"{k}{v}" if k.strip() else ''
        for k, v in lines
    )
    ax_stats.text(0.08, 0.92, f'PASO {step:03d}',
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
    units      = sim.units
    blue_all   = [u for u in units if u.team == 0]
    red_all    = [u for u in units if u.team == 1]
    blue_alive = [u for u in blue_all if u.alive]
    red_alive  = [u for u in red_all  if u.alive]

    winner      = sim.get_winner() or 'Draw'
    winner_es   = {'Blue': 'Azul', 'Red': 'Rojo', 'Draw': 'Empate'}
    winner_name = winner_es.get(winner, winner)
    wins        = sum(1 for r in opt_results if r['won'])
    total_iter  = len(opt_results)
    success_rt  = wins / total_iter if total_iter else 0.0
    weather_str = WEATHER_NAMES.get(sim.weather, str(sim.weather))

    def tc(team_units, utype):
        return sum(1 for u in team_units if u.unit_type == utype)

    arm_names = {
        UnitType.INFANTRY:       'Infanteria    ',
        UnitType.ARTILLERY:      'Artilleria    ',
        UnitType.CAVALRY:        'Caballeria    ',
        UnitType.COMMUNICATIONS: 'Comunicaciones',
        UnitType.ENGINEERING:    'Ingenieria    ',
        UnitType.LOGISTICS:      'Logistica     ',
    }

    sep = '=' * 64

    lines = [
        sep,
        '  SIMULACION TACTICA MILITAR - REPORTE ANALITICO',
        '  6 Armas del Ejercito de Bolivia',
        sep,
        '',
        f'  RESULTADO DE MISION : {winner_name.upper()}'
        f'{"  VICTORIA" if winner != "Draw" else ""}',
        f'  Condicion climatica : {weather_str}',
        '',
        '  OPTIMIZACION EVOLUTIVA  (Regla 1/5 de Rechenberg)',
        f'    Iteraciones ejecutadas : {total_iter}',
        f'    Ejecuciones exitosas   : {wins}',
        f'    Tasa de exito          : {success_rt:.1%}',
        '',
        '  PARAMETROS TACTICOS OPTIMIZADOS',
        f'    Agresividad            : {best_params.get("aggressiveness", 0):.3f}',
        f'    Busqueda de cobertura  : {best_params.get("cover_seeking", 0):.3f}',
        f'    Trabajo en equipo      : {best_params.get("teamwork", 0):.3f}',
        '',
        '  COMPOSICION DE FUERZAS  (6 Armas)',
        '    Azul (Atacante)',
    ]
    for ut, lbl in arm_names.items():
        lines.append(f'      {lbl} : {tc(blue_all, ut)}')
    lines += ['    Rojo (Defensor)']
    for ut, lbl in arm_names.items():
        lines.append(f'      {lbl} : {tc(red_all, ut)}')

    lines += [
        '',
        '  ESTADISTICAS DE BAJAS',
        '    Azul (Atacante)',
        f'      Unidades iniciales     : {len(blue_all)}',
        f'      Sobrevivientes         : {len(blue_alive)}',
        f'      Bajas                  : {len(blue_all)-len(blue_alive)}'
        f'  ({(len(blue_all)-len(blue_alive))/max(len(blue_all),1)*100:.1f}%)',
        '    Rojo (Defensor)',
        f'      Unidades iniciales     : {len(red_all)}',
        f'      Sobrevivientes         : {len(red_alive)}',
        f'      Bajas                  : {len(red_all)-len(red_alive)}'
        f'  ({(len(red_all)-len(red_alive))/max(len(red_all),1)*100:.1f}%)',
        '',
        '  CONSUMO DE RECURSOS (estado final)',
        f'    Municion prom. Azul    : {np.mean([u.ammo for u in blue_all]):.1f}',
        f'    Municion prom. Rojo    : {np.mean([u.ammo for u in red_all]):.1f}',
        f'    Moral prom. Azul       : {np.mean([u.morale for u in blue_alive] or [0]):.2f}',
        f'    Moral prom. Rojo       : {np.mean([u.morale for u in red_alive]  or [0]):.2f}',
        '',
        '  REGISTRO DE ITERACIONES',
    ]

    for r in opt_results:
        p   = r['params']
        tag = 'GANO ' if r['won'] else 'PERDIO'
        lines.append(
            f"    Iter {r['iteration']:2d}: {tag} | "
            f"Agres={p['aggressiveness']:.2f}  "
            f"Cob={p['cover_seeking']:.2f}  "
            f"Eq={p['teamwork']:.2f}  |  "
            f"Bajas Azul: {r['casualties']}"
        )

    lines += ['', sep]
    report = '\n'.join(lines)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)

    print('\n' + report)


# -- Main entry point ----------------------------------------------------------
def run_simulation(steps: int = 60, n_opt_iter: int = 10, mode: str = 'live',
                   weather: Weather = Weather.CLEAR, use_editor: bool = False):
    print('=' * 60)
    print('  Simulador Tactico Militar - Open House 2026')
    print('  6 Armas del Ejercito de Bolivia')
    print('=' * 60)

    loader  = MapLoader('assets/image.png', grid_width=80)
    terrain = loader.get_grid()
    print(f'\n  Grilla : {terrain.shape[0]} filas x {terrain.shape[1]} columnas')

    unique, counts = np.unique(terrain, return_counts=True)
    names_t = {0:'Vacio',1:'Bosque',2:'Obstaculo',3:'Objetivo',4:'Urbano',5:'Suministro'}
    for u, c in zip(unique, counts):
        print(f'    {names_t.get(int(u), u)}: {c} celdas ({c/terrain.size*100:.1f}%)')

    # Phase 1: Rechenberg optimization
    optimizer = RechenbergOptimizer(terrain, n_iterations=n_opt_iter, sim_steps=steps)
    best_params, _, opt_results = optimizer.optimize()

    # Phase 2: Final simulation with best params + chosen weather
    print(f'\n--- Simulacion final ({steps} pasos, clima: {WEATHER_NAMES[weather]}) ---')
    sim = CASimulator(terrain, params=best_params, weather=weather)

    def arm_count(team_units, utype):
        return sum(1 for u in team_units if u.unit_type == utype)

    blue_u = [u for u in sim.units if u.team == 0]
    red_u  = [u for u in sim.units if u.team == 1]
    for label, lst in [('Azul', blue_u), ('Rojo', red_u)]:
        inf = arm_count(lst, UnitType.INFANTRY)
        art = arm_count(lst, UnitType.ARTILLERY)
        cab = arm_count(lst, UnitType.CAVALRY)
        com = arm_count(lst, UnitType.COMMUNICATIONS)
        ing = arm_count(lst, UnitType.ENGINEERING)
        log = arm_count(lst, UnitType.LOGISTICS)
        print(f'  {label}: {len(lst)} u  '
              f'INF:{inf} ART:{art} CAB:{cab} COM:{com} ING:{ing} LOG:{log}')

    if mode == 'live':
        print('\n  Abriendo simulacion interactiva...')
        print('  ESPACIO=pausa  +/-=vel  R=reiniciar  H=calor  ESC=salir  S=captura\n')

        viz = PygameVisualizer(terrain, sim, fps=8,
                               title='Simulador Tactico Militar - Open House 2026')

        if use_editor:
            print('  [EDITOR] Configura el mapa, luego ENTER para iniciar\n')
            custom_units, chosen_weather = viz.run_editor()
            sim = CASimulator(terrain, params=best_params, weather=chosen_weather,
                              custom_units=custom_units)
            viz.sim = sim

        sim = viz.run(max_steps=steps)

    else:
        last_frame_path = None
        final_step      = steps - 1

        for i in range(steps):
            blue_grid, red_grid, stats = sim.step()
            save_path = os.path.join(OUTPUT_DIR, f'frame_{i:03d}.png')
            render_frame(terrain, blue_grid, red_grid, stats, i, save_path)
            last_frame_path = save_path

            if i % 5 == 0 or i == steps - 1:
                print(f'  Paso {i:3d} | Azul: {stats["blue_alive"]:3d} | '
                      f'Rojo: {stats["red_alive"]:3d} | '
                      f'Salud Azul: {stats["blue_avg_health"]:5.1f} | '
                      f'Salud Rojo: {stats["red_avg_health"]:5.1f}')

            winner = sim.get_winner()
            if winner is not None:
                wes = {'Blue': 'Azul', 'Red': 'Rojo', 'Draw': 'Empate'}
                print(f'\n  *** Mision terminada paso {i}: {wes.get(winner, winner)} gana! ***')
                final_step = i
                for j in range(i + 1, steps):
                    shutil.copy(last_frame_path,
                                os.path.join(OUTPUT_DIR, f'frame_{j:03d}.png'))
                break

        print(f'\n  Frames -> {OUTPUT_DIR}/ (frame_000 ... frame_{final_step:03d})')

    # Phase 3: Analytical report
    report_path = os.path.join(OUTPUT_DIR, 'reporte_tactico.txt')
    generate_report(sim, opt_results, best_params, report_path)
    print(f'  Reporte -> {report_path}')
    print('\n  Simulacion completa.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulador Tactico Militar de Automatas Celulares - Open House 2026')
    parser.add_argument('--mode', choices=['live', 'export'], default='live',
                        help='live = ventana Pygame (default), export = PNG frames')
    parser.add_argument('--steps', type=int, default=60,
                        help='Pasos de simulacion (default: 60)')
    parser.add_argument('--opt-iter', type=int, default=10,
                        help='Iteraciones Rechenberg (default: 10)')
    parser.add_argument('--weather',
                        choices=['clear','rain','fog','cold','heat'], default='clear',
                        help='Condicion climatica inicial')
    parser.add_argument('--editor', action='store_true',
                        help='Abrir editor de mapa antes de la simulacion')
    args = parser.parse_args()

    weather_map = {
        'clear': Weather.CLEAR, 'rain': Weather.RAIN, 'fog': Weather.FOG,
        'cold':  Weather.COLD,  'heat': Weather.HEAT,
    }
    run_simulation(
        steps=args.steps, n_opt_iter=args.opt_iter, mode=args.mode,
        weather=weather_map[args.weather], use_editor=args.editor,
    )
