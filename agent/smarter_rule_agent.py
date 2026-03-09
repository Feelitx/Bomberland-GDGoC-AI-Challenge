import random
from collections import deque

import numpy as np


class SmarterRuleAgent:
    """
    Actions:
    0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB
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

        alive_enemies = []
        for i, p in enumerate(players):
            if i != self.id and p[2] == 1:
                alive_enemies.append((int(p[0]), int(p[1])))

        occupied = {
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if p[2] == 1 and i != self.id
        }

        danger_soon, danger_now = self._danger_tiles(grid, bombs, default_radius=2)
        valid_actions = self._valid_actions(grid, my_pos, occupied)

        # 1) Escape if in immediate danger
        if my_pos in danger_now or my_pos in danger_soon:
            escape = self._move_to_nearest_safe(
                grid, my_pos, occupied, danger_soon, search_depth=8
            )
            if escape is not None:
                return escape
            safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_now]
            return random.choice(safe_moves) if safe_moves else 0

        # 2) Place bomb if enemy in blast line and can likely escape
        if bombs_left > 0 and self._can_hit_enemy_with_bomb(grid, my_pos, alive_enemies, bomb_radius):
            if self._can_escape_after_placing(grid, my_pos, occupied, danger_soon, bomb_radius):
                return 5

        # 3) Move toward nearest enemy while avoiding danger
        if alive_enemies:
            move = self._move_toward_enemy(grid, my_pos, alive_enemies, occupied, danger_soon)
            if move is not None:
                return move

        # 4) Fallback safe random walk
        safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0

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

    def _danger_tiles(self, grid, bombs, default_radius=2):
        danger_soon = set()
        danger_now = set()
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            if timer <= 0:
                continue
            blast = self._blast_tiles(grid, bx, by, default_radius)
            danger_soon |= blast
            if timer <= 1:
                danger_now |= blast
        return danger_soon, danger_now

    def _move_to_nearest_safe(self, grid, start, occupied, danger_soon, search_depth=8):
        q = deque([(start, 0, None)])
        seen = {start}
        while q:
            pos, d, first_action = q.popleft()
            if pos not in danger_soon and d > 0:
                return first_action
            if d >= search_depth:
                continue

            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                if a != 0 and (not self._passable(grid, nx, ny) or (nx, ny) in occupied):
                    continue
                npos = (nx, ny)
                if npos in seen:
                    continue
                seen.add(npos)
                q.append((npos, d + 1, a if first_action is None else first_action))
        return None

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

    def _can_hit_enemy_with_bomb(self, grid, my_pos, enemies, radius):
        mx, my = my_pos
        for ex, ey in enemies:
            if mx == ex and abs(ey - my) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
            if my == ey and abs(ex - mx) <= radius and self._line_clear(grid, my_pos, (ex, ey)):
                return True
        return False

    def _can_escape_after_placing(self, grid, my_pos, occupied, existing_danger, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        combined_danger = set(existing_danger) | my_blast
        action = self._move_to_nearest_safe(grid, my_pos, occupied, combined_danger, search_depth=6)
        return action is not None

    def _move_toward_enemy(self, grid, start, enemies, occupied, danger_soon):
        targets = set(enemies)
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if not self._passable(grid, nx, ny) or npos in occupied or npos in seen:
                    continue
                if npos in danger_soon:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None