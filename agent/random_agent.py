import random

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id
    
    def act(self, observation):
        return random.randint(0, 5)