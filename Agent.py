
import Grid

import random

visibilityRadius = 2

class Agent:

    def __init__(self, x,y, agent_id: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y

    def receiveObservation():
        pass

    def move(self):

        # 8 directions + stay
        moves = [(-1,-1), (0,-1), (1,-1),
                 (-1, 0), (0, 0), (1, 0),
                 (-1, 1), (0, 1), (1, 1)]

        # Generate candidate new positions (bounded)
        candidateMoves = []
        for dx, dy in moves:
            newX = min(max(self.x + dx, 0), Grid.gridWidth  - 1)
            newY = min(max(self.y + dy, 0), Grid.gridHeight - 1)
            candidateMoves.append((newX, newY))

        # Deduplicate in case clamping created duplicates, and shuffle to break ties fairly
        candidateMoves = list({(nx, ny) for (nx, ny) in candidateMoves})
        random.shuffle(candidateMoves)

        #pick random move   
        new_x, new_y = random.choice(candidateMoves)

        self.x , self.y = new_x, new_y
        return self.x, self.y