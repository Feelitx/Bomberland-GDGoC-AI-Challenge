import random
from collections import deque

import numpy as np


class GeniusRuleAgent:
    """
    Observation-heavy heuristic agent:
    - threat-aware movement (immediate vs near-future danger),
    - enemy pursuit with pathfinding,
    - bomb placement only when hit potential + escape path exists.
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, player_id: int):
        self.id = int(player_id)

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.id >= len(players) or players[self.id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        enemies = []
        occupied = set()
        for i, p in enumerate(players):
            if p[2] != 1:
                continue
            pos = (int(p[0]), int(p[1]))
            if i == self.id:
                continue
            enemies.append(pos)
            occupied.add(pos)

        danger_t1, danger_t3 = self._danger_sets(grid, bombs, default_radius=2)
        valid_actions = self._valid_actions(grid, my_pos, occupied)

        # Forced escape first
        if my_pos in danger_t1:
            escape = self._best_escape_action(grid, my_pos, occupied, danger_t1, danger_t3)
            return escape if escape is not None else 0

        # Score each action
        best_score = -10**9
        best_actions = []

        for action in valid_actions + ([5] if bombs_left > 0 else []):
            score = self._score_action(
                action=action,
                grid=grid,
                my_pos=my_pos,
                enemies=enemies,
                occupied=occupied,
                danger_t1=danger_t1,
                danger_t3=danger_t3,
                bomb_radius=bomb_radius,
            )
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions) if best_actions else 0

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] == 0

    def _valid_actions(self, grid, my_pos, occupied):
        actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                actions.append(a)
        return actions

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                if grid[x, y] == 1:
                    break
                tiles.add((x, y))
        return tiles

    def _danger_sets(self, grid, bombs, default_radius=2):
        danger_t1 = set()  # timer <= 1
        danger_t3 = set()  # timer <= 3
        for b in bombs:
            bx, by, t = int(b[0]), int(b[1]), int(b[2])
            if t <= 0:
                continue
            blast = self._blast_tiles(grid, bx, by, default_radius)
            if t <= 1:
                danger_t1 |= blast
            if t <= 3:
                danger_t3 |= blast
        return danger_t1, danger_t3

    def _line_clear(self, grid, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if grid[ax, y] == 1:
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if grid[x, ay] == 1:
                    return False
            return True
        return False

    def _dist_to_nearest_enemy(self, start, enemies):
        if not enemies:
            return 999
        return min(abs(start[0] - ex) + abs(start[1] - ey) for ex, ey in enemies)

    def _nearest_enemy_path_dist(self, grid, start, enemies, occupied, blocked=None, max_depth=30):
        if not enemies:
            return 999
        targets = set(enemies)
        blocked = blocked or set()

        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if pos in targets:
                return d
            if d >= max_depth:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen or npos in blocked:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in occupied and npos not in targets:
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return 999

    def _has_escape(self, grid, start, occupied, blocked, depth=7):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if pos not in blocked and d > 0:
                return True
            if d >= depth:
                continue
            for a in [0, 1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if a != 0:
                    if not self._passable(grid, nx, ny) or npos in occupied:
                        continue
                seen.add(npos)
                q.append((npos, d + 1))
        return False

    def _best_escape_action(self, grid, my_pos, occupied, danger_t1, danger_t3):
        best_action = None
        best_val = -10**9
        for a in self._valid_actions(grid, my_pos, occupied):
            npos = self._next_pos(my_pos, a)
            if npos in danger_t1:
                continue
            val = 0
            if npos not in danger_t3:
                val += 5
            val += self._open_neighbors(grid, npos, occupied)
            if val > best_val:
                best_val = val
                best_action = a
        return best_action

    def _open_neighbors(self, grid, pos, occupied):
        cnt = 0
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                cnt += 1
        return cnt

    def _can_bomb_hit_enemy(self, grid, my_pos, enemies, radius):
        mx, my = my_pos
        for ex, ey in enemies:
            if mx == ex and abs(ey - my) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
            if my == ey and abs(ex - mx) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
        return False

    def _score_action(
        self,
        action,
        grid,
        my_pos,
        enemies,
        occupied,
        danger_t1,
        danger_t3,
        bomb_radius,
    ):
        # Bomb action: high reward only when tactical + survivable
        if action == 5:
            if not self._can_bomb_hit_enemy(grid, my_pos, enemies, bomb_radius):
                return -50
            my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
            blocked = set(danger_t3) | my_blast
            if not self._has_escape(grid, my_pos, occupied, blocked, depth=7):
                return -80
            return 45 + (8 if my_pos not in danger_t3 else 0)

        # Movement/stop action
        npos = self._next_pos(my_pos, action)

        # Safety dominates
        if npos in danger_t1:
            return -100
        score = 0
        if npos in danger_t3:
            score -= 20
        else:
            score += 8

        # Prefer tiles with options (avoid dead-ends)
        score += 2 * self._open_neighbors(grid, npos, occupied)

        # Enemy pressure
        path_before = self._nearest_enemy_path_dist(grid, my_pos, enemies, occupied, blocked=danger_t1)
        path_after = self._nearest_enemy_path_dist(grid, npos, enemies, occupied, blocked=danger_t1)
        score += (path_before - path_after) * 3

        # Mild inertia against stop
        if action == 0:
            score -= 1

        return score