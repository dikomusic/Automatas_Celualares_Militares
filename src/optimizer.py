import numpy as np
from src.engine import CASimulator


class RechenbergOptimizer:
    """
    Rechenberg 1/5 success rule for tactical parameter optimization.

    Runs `n_iterations` independent simulations with perturbed parameters.
    If the success rate exceeds 1/5, sigma is increased (explore more);
    if it falls below 1/5, sigma is decreased (exploit neighbourhood).
    The best parameter set (most Blue wins) is returned.
    """

    SIGMA_FACTOR = 1.22   # Rechenberg adjustment factor

    def __init__(self, terrain: np.ndarray, n_iterations: int = 10,
                 sim_steps: int = 30):
        self.terrain       = terrain
        self.n_iterations  = n_iterations
        self.sim_steps     = sim_steps
        self.sigma         = 0.12

        # Starting point for parameter search
        self.params = {
            'aggressiveness': 0.65,
            'cover_seeking':  0.50,
            'teamwork':       0.30,
        }

        self.results: list[dict] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _perturb(self) -> dict:
        """Add zero-mean Gaussian noise with current sigma to each parameter."""
        return {
            k: float(np.clip(v + np.random.normal(0.0, self.sigma), 0.05, 0.95))
            for k, v in self.params.items()
        }

    def _evaluate(self, params: dict) -> tuple:
        """Run one full simulation; return (won: bool, sim: CASimulator)."""
        sim = CASimulator(self.terrain, params=params)
        winner = None
        for _ in range(self.sim_steps):
            sim.step()
            winner = sim.get_winner()
            if winner is not None:
                break
        if winner is None:
            winner = sim.get_winner()
        return winner == 'Blue', sim

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(self) -> tuple:
        """
        Run the Rechenberg loop.

        Returns
        -------
        best_params : dict
        best_sim    : CASimulator   (from the winning run, or last run)
        results     : list[dict]    (one entry per iteration)
        """
        print("\n--- Rechenberg Evolutionary Optimization (1/5 Rule) ---")
        print(f"  Iterations: {self.n_iterations}  |  Steps/run: {self.sim_steps}")
        print(f"  Initial sigma: {self.sigma:.4f}\n")

        successes   = 0
        best_params = dict(self.params)
        best_sim    = None

        for i in range(self.n_iterations):
            candidate = self._perturb()
            won, sim  = self._evaluate(candidate)

            blue_start = sum(1 for u in sim.units if u.team == 0)
            blue_end   = sum(1 for u in sim.units if u.alive and u.team == 0)
            casualties = blue_start - blue_end

            entry = {
                'iteration':  i + 1,
                'params':     candidate,
                'won':        won,
                'casualties': casualties,
                'sim':        sim,
            }
            self.results.append(entry)

            if won:
                successes   += 1
                best_params  = candidate
                best_sim     = sim

            tag = 'WIN ' if won else 'LOSS'
            print(f"  Iter {i+1:2d}: {tag} | "
                  f"Aggr={candidate['aggressiveness']:.2f}  "
                  f"Cover={candidate['cover_seeking']:.2f}  "
                  f"Team={candidate['teamwork']:.2f}  |  "
                  f"Blue casualties: {casualties}")

        # ── Apply 1/5 rule ────────────────────────────────────────────────────
        success_rate = successes / self.n_iterations
        if success_rate > 0.20:
            self.sigma *= self.SIGMA_FACTOR   # Widen search
        else:
            self.sigma /= self.SIGMA_FACTOR   # Narrow search

        self.params = best_params

        print(f"\n  Success rate : {success_rate:.1%}")
        print(f"  Adjusted sigma: {self.sigma:.4f}")
        print(f"  Best params  : {best_params}")

        # If no run was won, return the sim from the last iteration
        if best_sim is None:
            best_sim = self.results[-1]['sim']

        return best_params, best_sim, self.results
