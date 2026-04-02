import os
import sys
import matplotlib
# Use 'Agg' backend to save files without needing a GUI window (X11/Wayland)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Programmatically add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import your custom modules
from src.mapper import MapLoader
from src.engine import CASimulator

# Ensure the output directory exists
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_simulation(steps=30):
    print("--- Military CA Simulation Starting ---")

    # 1. Initialize the Map Loader
    # Ensure assets/map_test.png exists in your folder
    try:
        loader = MapLoader('assets/map_test.png', grid_width=150)
        terrain = loader.get_grid()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Initialize the Simulator with the terrain
    sim = CASimulator(terrain)

    print(f"Grid Size: {terrain.shape[0]}x{terrain.shape[1]}")
    print(f"Simulating {steps} steps...")

    for i in range(steps):
        # Calculate the next state
        unit_state = sim.step()

        # 3. Create the Visualization
        plt.figure(figsize=(10, 8))

        # Draw Terrain (0=Black, 1=Green, 2=Grey, 3=Red)
        # We use 'terrain' or 'nipy_spectral' for distinct military-style colors
        plt.imshow(terrain, cmap='nipy_spectral', alpha=0.6)

        # Overlay Units (Blue)
        # We mask the '0' values so the background terrain shows through
        masked_units = np.ma.masked_where(unit_state == 0, unit_state)
        plt.imshow(masked_units, cmap='winter', alpha=0.9)

        plt.title(f"Military Operation Simulation - Step {i:03d}")
        plt.axis('off') # Remove axis for a cleaner "map" look

        # Save the frame
        save_path = os.path.join(OUTPUT_DIR, f'frame_{i:03d}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close() # Free up memory

        if i % 5 == 0:
            print(f"Progress: Step {i} saved to {save_path}")

    print(f"--- Simulation Complete. Check the '{OUTPUT_DIR}' folder for frames. ---")

if __name__ == "__main__":
    run_simulation()
