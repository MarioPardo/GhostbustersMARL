import random
from Constants import *

gridWidth = 30
gridHeight = 30

class Ghost:
    """
    Minimal ghost entity for a MARL ghostbusters environment.
    """
    def __init__(self, x,y,movementProb=1.0):
        self.x = x
        self.y = y
        self.movementProb = movementProb  # Probability of moving each turn

        self.avoidRadius = 3


    def move(self, agentCoords):
        """
        Move randomly, avoiding agents within avoidRadius (Chebyshev distance).
        """
        
        movementProb = self.movementProb
        if movementProb <= 0:
            return self.x, self.y  #stay put

        #Decrease movement probability based on how surrounded the ghost is
        movementPenalty = min(1, 0.2 * self.GetSurroundedCount(agentCoords))
        movementProb = movementProb * movementPenalty
        movementProb = max(movementProb, 0.0)
        if random.random() > movementProb:
            return self.x, self.y  #stay put


        # 8 directions + stay
        moves = [(-1,-1), (0,-1), (1,-1),
                 (-1, 0), (0, 0), (1, 0),
                 (-1, 1), (0, 1), (1, 1)]

        # Generate candidate new positions (bounded)
        candidateMoves = []
        for dx, dy in moves:
            newX = min(max(self.x + dx, 0), gridWidth  - 1)
            newY = min(max(self.y + dy, 0), gridHeight - 1)
            candidateMoves.append((newX, newY))

        # Deduplicate in case clamping created duplicates, and shuffle to break ties fairly
        candidateMoves = list({(nx, ny) for (nx, ny) in candidateMoves})
        random.shuffle(candidateMoves)

        def min_dist_to_agents(p):
            return min((cheb_dist(p, a) for a in agentCoords), default=float("inf"))

        # Sort by descending distance to the nearest agent (maximize separation)
        candidateMoves.sort(key=lambda p: min_dist_to_agents(p), reverse=True)
        occupied = set(agentCoords)

        #If the closest agent is far enough, stay put
        if min_dist_to_agents((self.x, self.y)) > self.avoidRadius:
            return self.x, self.y  #stay put
        
        # Else, we avoid agents.
        for nx, ny in candidateMoves:  # Try moves best to worst
            if (nx, ny) not in occupied:
                new_x, new_y = nx, ny
                break
        else: #if all occupied, stay put
            new_x, new_y = self.x, self.y

        self.x, self.y = new_x, new_y
        return self.x, self.y
    

    def GetSurroundedCount(self, agentCoords):
        """Update the counter for how long the ghost has been surrounded."""
        ghost_x, ghost_y = self.x,self.y
        surroundCounter = sum(
            cheb_dist((a[0], a[1]), (ghost_x, ghost_y)) <= SURROUND_RADIUS
            for a in agentCoords
        )

        return surroundCounter




    