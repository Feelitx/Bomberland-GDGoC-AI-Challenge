import random
import numpy as np

class SimpleRuleAgent:
    def __init__(self, player_id):
        self.player_id = player_id
    
    def act(self, observation):
        my_pos = observation["players"][self.player_id]
        bombs = observation["bombs"]
        
        for b in bombs:
            if b[2] <= 1: # bomb about to explode -> run away
                if abs(b[0] - my_pos[0]) + abs(b[1] - my_pos[1]) <= 1:
                    return random.randint(1, 4)
        
        # if adjacent to opponent -> place bomb
        for i, p in enumerate(observation["players"]):
            if i != self.player_id and p[2] == 1: # alive
                if abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]) == 1:
                    return 5
        # otherwise, move randomly
        return random.randint(1, 4)