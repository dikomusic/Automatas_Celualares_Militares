import numpy as np
from src.entities.unit import Unit
from src.environment.grid import TacticalGrid

def resolve_engagement(attacker: Unit, defender: Unit, grid: TacticalGrid):
    """
    Resolves a combat exchange between two units.
    
    Factors in distance, morale, terrain cover, and faction traits.
    """
    if attacker.ammo <= 0:
        attacker.update_morale(-0.02)
        return

    # 1. Distance Calculation (Chebyshev)
    dist = max(abs(attacker.x - defender.x), abs(attacker.y - defender.y))
    fire_range = getattr(attacker, 'vision_radius', 12)

    if dist > fire_range:
        return

    # 2. Ammo Consumption
    attacker.ammo -= 1

    # 3. Hit Probability Logic
    # Base 60% chance, modified by morale, distance, and teamwork
    base_hit = 0.60
    dist_penalty = (1.0 - dist / fire_range)
    teamwork_bonus = 1.0 + (attacker.teamwork * 0.2) # Simple synergy buff
    
    hit_prob = base_hit * attacker.morale * dist_penalty * teamwork_bonus

    if np.random.random() < hit_prob:
        # 4. Damage Calculation (Rule 1: Cover Bonus)
        cover = grid.get_cover_bonus(defender.x, defender.y)
        base_damage = 25.0
        
        # Damage mitigation from terrain
        damage = base_damage * (1.0 - cover) * np.random.uniform(0.8, 1.2)
        
        # Apply Damage & Morale impact
        defender.take_damage(damage)
        attacker.update_morale(0.02) # Success boost
        
        # Rule 6: Suppression (Pinned State)
        if damage > 15.0 or np.random.random() < 0.3:
            defender.pinned_timer = 2 # Pinned for 2 turns
    else:
        # Misses still suppress slightly
        defender.update_morale(-0.02)
