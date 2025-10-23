import random
from Constants import cheb_dist

gridWidth = 30
gridHeight = 30

class Ghost:
    """
    Minimal ghost entity for a MARL ghostbusters environment.
    """
    def __init__(self, x,y):
        self.x = x
        self.y = y

        self.Health = 90 #maybe use later on
        self.avoidRadius = 5


    def move(self, agentCoords):
        """
        Move randomly, avoiding agents within avoidRadius (Chebyshev distance).
        """
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

        #If the closest agent is far enough, just move randomly
        if min_dist_to_agents((self.x, self.y)) > self.avoidRadius:
            new_x, new_y = random.choice(candidateMoves)
            self.x, self.y = new_x, new_y
            return self.x, self.y

        # Else, we avoid agents.
        # Try moves best to worst
        for nx, ny in candidateMoves:
            if (nx, ny) not in occupied:
                new_x, new_y = nx, ny
                break
        else: #if all occupied, stay put
            new_x, new_y = self.x, self.y

        self.x, self.y = new_x, new_y
        return self.x, self.y



    