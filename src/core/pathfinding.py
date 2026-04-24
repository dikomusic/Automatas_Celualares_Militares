import heapq
from typing import List, Tuple, Optional

class Pathfinder:
    """
    Spatial Intelligence Module for Tactical Agents.
    """

    def __init__(self):
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (1.414 - 2) * min(dx, dy)

    def get_optimal_path(self, start: Tuple[int, int], target: Tuple[int, int], grid) -> List[Tuple[int, int]]:
        if not grid.is_passable(target[0], target[1]):
            return []

        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == target: break

            for dx, dy in self.directions:
                next_node = (current[0] + dx, current[1] + dy)
                if not grid.is_passable(next_node[0], next_node[1]): continue

                move_weight = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                new_cost = current_cost + (grid.get_cost(next_node[0], next_node[1]) * move_weight)

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self._heuristic(next_node, target)
                    heapq.heappush(frontier, (priority, new_cost, next_node))
                    came_from[next_node] = current

        if target not in came_from: return []
        path, curr = [], target
        while curr is not None:
            path.append(curr); curr = came_from[curr]
        path.reverse()
        return path

    def find_nearest_supply(self, start: Tuple[int, int], supply_depots: List[Tuple[int, int]], grid) -> Optional[Tuple[int, int]]:
        if not supply_depots: return None
        depot_set = set(supply_depots)
        frontier = [(0, start)]
        visited_costs = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current in depot_set: return current
            if current_cost > visited_costs.get(current, float('inf')): continue

            for dx, dy in self.directions:
                next_node = (current[0] + dx, current[1] + dy)
                if not grid.is_passable(next_node[0], next_node[1]): continue

                move_weight = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                new_cost = current_cost + (grid.get_cost(next_node[0], next_node[1]) * move_weight)

                if next_node not in visited_costs or new_cost < visited_costs[next_node]:
                    visited_costs[next_node] = new_cost
                    heapq.heappush(frontier, (new_cost, next_node))
        return None
