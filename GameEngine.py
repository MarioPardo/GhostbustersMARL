

from Grid import Grid
from Agent import Agent
from Ghost import Ghost

import pygame

presetAgentCoords = [(5,3), (15,5), (20,4), (25,3)]
presetGhostCoords = (15,25)
presetExtractionPoint = (25,29)

class GameEngine:

    def __init__(self, extractionX: int, extractionY: int):
        
        self.grid = Grid(extractionX, extractionY)

        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=False)
        self.grid.setEntities(self.agentCoords, self.ghostCoords) 

        #Create Agents, Ghost
        self.agents = []
        for i, (ax, ay) in enumerate(self.agentCoords):
            agent = Agent(ax, ay, agent_id=i)
            self.agents.append(agent)

        self.ghost = Ghost(self.ghostCoords[0], self.ghostCoords[1])




    def SpawnEntities(self, randomized: bool):
        """
        Spawn agents and ghost at random non-overlapping locations
        """

        if not randomized:
            agentCoords = presetAgentCoords
            ghostCoords = presetGhostCoords
        else:
            #TODO make sure not overlap, and ghost is not visible to any agent initially
            pass

        return agentCoords, ghostCoords
    

    def step(self):
        """
        Advance the game state by one step.
        """

        #Move Ghost
        self.ghostCoords = self.ghost.move(self.agentCoords)

        #Move Agents
        for agents in self.agents:
            agents.move()
        self.agentCoords = [(agent.x, agent.y) for agent in self.agents]


        #Update Grid
        self.grid.setEntities(self.agentCoords, self.ghostCoords)




# ----------- Small demo loop -----------
if __name__ == "__main__":

    Game = GameEngine(presetExtractionPoint[0], presetExtractionPoint[1])
    Game.grid.init_display(cell_size=20)

    clock = pygame.time.Clock()
    running = True
    while running:
        # basic event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        Game.step()
        Game.grid.draw_grid(show_grid_lines=True)
        clock.tick(30)

    Game.grid.close_display()

