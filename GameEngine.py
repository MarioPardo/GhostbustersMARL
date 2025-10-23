

from Grid import Grid
from Agent import Agent
from Ghost import Ghost

from Constants import *

import pygame
import numpy as np

presetAgentCoords = [(5,3), (15,5), (20,4), (25,3)]
presetGhostCoords = (15,25)
presetExtractionPoint = (25,29)

class GameEngine:

    def __init__(self, extractionX: int, extractionY: int, episodeLimit: int):
        
        self.grid = Grid(extractionX, extractionY)

        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=False)
        self.grid.setEntities(self.agentCoords, self.ghostCoords) 

        #Create Agents, Ghost
        self.agents = []
        for i, (ax, ay) in enumerate(self.agentCoords):
            agent = Agent(ax, ay, agent_id=i)
            self.agents.append(agent)

        self.ghost = Ghost(self.ghostCoords[0], self.ghostCoords[1])


        self.episodeLimit = episodeLimit

        self.timeToKill = TIME_TO_KILL
        self.surroundRadius = SURROUND_RADIUS

        self.Time = 0
        self.holdCounter = 0 # how long agents have been surrounding ghost

    def reset(self, randomized = False):
        """
        Reset the game state for a new episode.
        """

        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=randomized)
        self.grid.setEntities(self.agentCoords, self.ghostCoords) 

        #Reset Agents, Ghost
        for i, (ax, ay) in enumerate(self.agentCoords):
            self.agents[i].x = ax
            self.agents[i].y = ay

        self.ghost.x = self.ghostCoords[0]
        self.ghost.y = self.ghostCoords[1]

        self.holdCounter = 0
        self.Time = 0

        return self.full_obs_helper() #for debugging


    def SpawnEntities(self, randomized: bool):
        """
        Spawn agents and ghost at random non-overlapping locations
        """

        if not randomized:
            return presetAgentCoords, presetGhostCoords

        else:
            #TODO make sure not overlap, and ghost is not visible to any agent initially
            return [], None


    def simpleStep(self):
        """
        Advance the game state by one step.
        """

        #Move Ghost
        self.ghostCoords = self.ghost.move(self.agentCoords)

        #Move Agents
        for agents in self.agents:
            agents.randMove()
        self.agentCoords = [(agent.x, agent.y) for agent in self.agents]


        #Update Grid
        self.grid.setEntities(self.agentCoords, self.ghostCoords)

    
    def step(self, actions: list[int]):

        #Apply agent moves
        for a, act_id in zip(self.agents, actions):
            a.apply_action(act_id)
        self.agentCoords = [(a.x, a.y) for a in self.agents]

        #Move Ghost
        self.ghostCoords = self.ghost.move(self.agentCoords)

        #Update Grid #rending purposes
        self.grid.setEntities(self.agentCoords, self.ghostCoords)

        #reward & termination
        self.Time += 1
        reward, terminated = self.globalReward()

        info = {"t": self.Time, "hold": self._hold_counter}
        return reward, terminated, info

    def globalReward(self):
        reward = -0.01

        if self.isExtracting():
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        terminated = False
        if self._hold_counter >= self.timeToKill:
            reward += 50.0
            terminated = True
        elif self.Time >= self.episodeLimit:
            terminated = True

        return reward, terminated


    def isExtracting(self):

        if self.ghost.x != self.grid.extraction_point[0] or self.ghost.y != self.grid.extraction_point[1]:
            return False
        
        surroundCount = sum(cheb_dist((agent.x, agent.y), (self.ghost.x, self.ghost.y)) <= self.surroundRadius for agent in self.agents)
        return surroundCount == len(self.agents)


    def full_obs_helper(self):
        """Full-obs helper (normalized coords in [0,1]). Returns list per-agent."""
        W, H = self.grid.width, self.grid.height
        gx, gy = self.ghostCoords
        ex, ey = self.grid.extraction_point
        obs = []
        for a in self.agents:
            vec = np.array([
                a.x/(W-1), a.y/(H-1),
                gx/(W-1), gy/(H-1),
                ex/(W-1), ey/(H-1),
            ], dtype=np.float32)
            obs.append(vec)
        return obs

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

        Game.simpleStep()
        Game.grid.draw_grid(show_grid_lines=True)
        clock.tick(30)

    Game.grid.close_display()

