

from Grid import Grid
from Agent import Agent
from Ghost import Ghost

from Constants import *

import pygame
import numpy as np

numAgents = 3

presetAgentCoords = [(5,3), (15,5), (20,4)]
presetGhostCoords = (15,25)

class GameEngine:

    def __init__(self, extractionTL_x: int, extractionTL_y: int, extractionBR_x:int, extractionBR_y, episodeLimit: int, reward_cfg, spawn_radius: int, ghost_move_prob: float):
        
        self.grid = Grid(extractionTL_x, extractionTL_y, extractionBR_x, extractionBR_y)
        
        self.spawn_radius = spawn_radius  # For curriculum learning
        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=True, spawn_radius=spawn_radius)
        self.grid.setEntities(self.agentCoords, self.ghostCoords) 

        #Create Agents, Ghost
        self.agents = []
        for i, (ax, ay) in enumerate(self.agentCoords):
            agent = Agent(ax, ay, agent_id=i)
            self.agents.append(agent)

        self.ghost = Ghost(self.ghostCoords[0], self.ghostCoords[1], movementProb=ghost_move_prob)


        self.episodeLimit = episodeLimit

        # Set up rewards
        self.reward_cfg = reward_cfg
        self.lambda_ghost_dist_to_extraction = reward_cfg.get("lambda_ghost_dist_to_extraction", 0.05)
        self.lambda_agent_dist_to_ghost = reward_cfg.get("lambda_agent_dist_to_ghost", 0.02)
        self.reward_surrounded = reward_cfg.get("reward_surrounded", 20)
        self.lambda_quadrant_coverage = reward_cfg.get("lambda_quadrant_coverage", 2)
        self.reward_new_surround = reward_cfg.get("reward_new_surround", 5)
        self.penalty_formation_break = reward_cfg.get("penalty_formation_break", 5.0)
        self.reward_full_surround = reward_cfg.get("reward_full_surround", 30.0)
        self.win_reward = reward_cfg.get("reward_kill", 1000)

        self.timeToKill = TIME_TO_KILL
        self.surroundRadius = SURROUND_RADIUS

        self.Time = 0
        self.holdCounter = 0 # how long agents have been surrounding ghost
        self.surroundCounter = 0

    def reset(self, randomized = True):
        """
        Reset the game state for a new episode.
        """

        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=randomized, spawn_radius=self.spawn_radius)
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


    def SpawnEntities(self, randomized: bool, spawn_radius: int = None):
        """
        Spawn agents and ghost at random non-overlapping locations.
        
        Args:
            randomized: If False, use preset positions
            spawn_radius: If provided, spawn ghost within this radius of at least one agent (curriculum)
        """

        if not randomized:
            return presetAgentCoords, presetGhostCoords

        else:
            # First spawn agents 
            agentSpawnpoints = []
            for i in range(numAgents):
                while True:
                    ax = np.random.randint(0, self.grid.width)
                    ay = np.random.randint(0, self.grid.height)
                    
                    if (ax, ay) not in agentSpawnpoints:
                        agentSpawnpoints.append((ax, ay))
                        break
            
            # Then spawn ghost based on spawn_radius
            if spawn_radius is not None:
                # CURRICULUM: Ghost within spawn_radius of at least one agent
                attempts = 0
                while attempts < 100:
                    # Pick a random agent to spawn near
                    target_agent = agentSpawnpoints[np.random.randint(0, numAgents)]
                    
                    # Random offset within spawn_radius
                    dx = np.random.randint(-spawn_radius, spawn_radius + 1)
                    dy = np.random.randint(-spawn_radius, spawn_radius + 1)
                    gx = np.clip(target_agent[0] + dx, 0, self.grid.width - 1)
                    gy = np.clip(target_agent[1] + dy, 0, self.grid.height - 1)
                    
                    if (gx, gy) not in agentSpawnpoints and not self.isExtracting((gx, gy)):
                        ghostSpawnpoint = (gx, gy)
                        break
                    attempts += 1
                
                if attempts >= 100:  # Fallback
                    while True:
                        gx = np.random.randint(0, self.grid.width)
                        gy = np.random.randint(0, self.grid.height)
                        if (gx, gy) not in agentSpawnpoints and not self.isExtracting((gx, gy)):
                            ghostSpawnpoint = (gx, gy)
                            break
            else:
                # NORMAL: Ghost at least VISIBILITY_RADIUS away from every agent
                while True:
                    gx = np.random.randint(0, self.grid.width)
                    gy = np.random.randint(0, self.grid.height)
                    
                    if (gx, gy) not in agentSpawnpoints and not self.isExtracting((gx, gy)):
                        # Check distance from all agents
                        all_far_enough = all(
                            cheb_dist((gx, gy), agent_pos) > VISIBILITY_RADIUS 
                            for agent_pos in agentSpawnpoints
                        )
                        if all_far_enough:
                            ghostSpawnpoint = (gx, gy)
                            break

            return agentSpawnpoints, ghostSpawnpoint


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


    def getAllAvailableActions(self):
        """
        Return a list of length n_agents, each entry a length-N_ACTIONS binary mask of legal actions.
        """
        
        i = 0
        avail_actions = []
        for a in self.agents:
            mask = self.getValidAgentActions(i)
            i+=1
            avail_actions.append(mask)
        return avail_actions
    
    def getValidAgentActions(self, i:int) -> np.ndarray:
        """
        Return a length-N_ACTIONS binary mask of legal actions for agent i,
        disallowing moves OOB or into occupied cells (other agents, ghost).
        """
        W, H = self.grid.width, self.grid.height

        ax, ay = self.agents[i].x, self.agents[i].y

        # cells occupied by other agents (exclude self)
        occupied = {(a.x, a.y) for k, a in enumerate(self.agents) if k != i}

        occupied.add(self.ghostCoords)

        mask = np.zeros((N_ACTIONS,), dtype=np.int32)

        for a_idx, (dx, dy) in enumerate(ACTION_TO_DELTA):
            nx, ny = ax + dx, ay + dy

            # Always allow "stay" (dx=0,dy=0)
            if dx == 0 and dy == 0:
                mask[a_idx] = 1
                continue

            # bounds check
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            # occupancy check
            if (nx, ny) in occupied:
                continue

            mask[a_idx] = 1

        return mask


    def step(self, actions: list[int]):


        ##Info about previous state used for calculating rewards
        #Previous distance ghost to extraction area
        prev_d_ghost_extract = cheb_dist(self.grid.extraction_point_center, self.ghostCoords)

        #Previous distance agents to ghost
        prev_d_ghost_agents = [cheb_dist((a.x, a.y), (self.ghost.x, self.ghost.y)) for a in self.agents]

        prev_surround_count = self.ghost.GetSurroundedCount(self.agentCoords)

        ###

        #Move Ghost FIRST (before agents) so agents can predict consequences
        self.ghostCoords = self.ghost.move(self.agentCoords)

        #Apply agent moves
        for a, act_id in zip(self.agents, actions):
            a.apply_action(act_id)
        self.agentCoords = [(a.x, a.y) for a in self.agents]

        self.surroundCounter = self.ghost.GetSurroundedCount(self.agentCoords)
        if(self.surroundCounter >= numAgents):
            self.grid.setSurrendered(True)
        else:
            self.grid.setSurrendered(False)

        #Update Grid #rending purposes
        self.grid.setEntities(self.agentCoords, self.ghostCoords)

        #reward & termination
        self.Time += 1
        reward, terminated, sucess = self.globalReward(prev_d_ghost_extract, prev_d_ghost_agents,prev_surround_count)

        info = {"t": self.Time, "hold": self.holdCounter, "success": sucess}
        return reward, terminated, info

    def globalReward(self, prev_d_ghost_extract, prev_d_ghost_agents, prev_surround_count):
        reward = -0.01  # Tiny time penalty to encourage efficiency 
        terminated = False
        success = False

        self.calculateUsefulMetrics

        if self.surroundCounter >= numAgents:
            self.holdCounter += 1
        else:
            self.holdCounter = 0

        # End conditions
        if self.holdCounter >= self.timeToKill:
            reward += self.win_reward 
            terminated = True
            success = True
        elif self.Time >= self.episodeLimit:
            terminated = True


        if terminated:
            return reward, terminated, success
        

        # Reward for getting agents closer to ghost
        reward += self.AgentToGhostDist_Reward(prev_d_ghost_agents)

        #Reward for surrounding ghost
        reward += self.Surround_Reward(prev_surround_count)
        
        # Reward for spreading around ghost at different angles
        reward += self.QuadrantCoverage_Reward()    


        return reward, terminated, success

    def AgentClumpingPenalty(self):
        penalty = 0.0
        CLUMP_DISTANCE = 2
        
        # Only penalize if not currently surrounding ghost
        if self.surroundCounter >= numAgents - 1:
            return 0.0  # Allow clustering when capturing
        
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                dist = cheb_dist((self.agents[i].x, self.agents[i].y), 
                                 (self.agents[j].x, self.agents[j].y))
                if dist <= CLUMP_DISTANCE:
                    penalty -= 0.1  # Reduced penalty
        
        return penalty

    
    def AgentToGhostDist_Reward(self, prev_agent_ghost_d):
        reward = 0.0

        # Closer to ghost = higher reward every step
        # Use Chebyshev distance to match surround logic!
        max_dist = max(self.grid.width - 1, self.grid.height - 1)
        
        for a in self.agents:
            dist = cheb_dist((a.x, a.y), self.ghostCoords)
            norm_dist = dist / max_dist
            # STRONGER exponential: cube instead of square for aggressive pull when close
            reward += self.lambda_agent_dist_to_ghost * ((1.0 - norm_dist) ** 3) * 3.0

        return reward
    
    def Surround_Reward(self, prev_surround_count):
        reward = 0.0
        new_surround_count = self.ghost.GetSurroundedCount(self.agentCoords)

        # Progressive surround rewards - BIG bonus for entering radius
        if new_surround_count > prev_surround_count:
            reward += self.reward_new_surround * (new_surround_count - prev_surround_count) * 2.0
        
        # PENALTY: Breaking formation (agents leaving surround) - REDUCED to encourage commitment
        if new_surround_count < prev_surround_count:
            penalty = self.penalty_formation_break * (prev_surround_count - new_surround_count) * 0.5
            reward -= penalty
        
        
        # Keeping full surround
        if new_surround_count >= numAgents:
            reward += self.reward_full_surround

        return reward

    def QuadrantCoverage_Reward(self):
        """
        Reward agents for surrounding ghost from different cardinal directions.
        Checks if agents are positioned left, right, above, or below the ghost.
        Rewards scale with proximity - closer agents get higher rewards.
        Always rewards coverage, not just when within surround radius.
        """
        
        gx, gy = self.ghostCoords
        directions = [False, False, False, False]  # Left, Right, Above, Below
        direction_distances = [float('inf'), float('inf'), float('inf'), float('inf')]  # Track closest agent per direction
        max_dist = max(self.grid.width, self.grid.height) - 1
        
        for ax, ay in self.agentCoords:
            dx = ax - gx
            dy = ay - gy
            dist = cheb_dist((ax, ay), (gx, gy))
            
            # Determine primary direction (using dominance of dx vs dy)
            if abs(dx) > abs(dy):  # Horizontal dominance
                if dx > 0:
                    directions[1] = True  # Right
                    direction_distances[1] = min(direction_distances[1], dist)
                else:
                    directions[0] = True  # Left
                    direction_distances[0] = min(direction_distances[0], dist)
            else:  # Vertical dominance
                if dy > 0:
                    directions[2] = True  # Above
                    direction_distances[2] = min(direction_distances[2], dist)
                else:
                    directions[3] = True  # Below
                    direction_distances[3] = min(direction_distances[3], dist)
        
        # Calculate reward: base reward per direction, scaled by proximity
        total_reward = 0.0
        for i, covered in enumerate(directions):
            if covered:
                # EXPONENTIAL proximity: reward grows rapidly as agents get very close
                normalized_dist = direction_distances[i] / max_dist
                proximity_multiplier = (2.0 - normalized_dist) ** 2  
                
                total_reward += self.lambda_quadrant_coverage * proximity_multiplier
        
        return total_reward

    def calculateUsefulMetrics(self):
        
        # Average Distance Agents to Ghost
        total_dist = sum(cheb_dist((a.x, a.y), self.ghostCoords) for a in self.agents)
        self.avgDistToGhost = total_dist / len(self.agents)
        


        return

    def isExtracting(self, ghostCoords):
        (x1, y1) = self.grid.extraction_area_tl
        (x2, y2) = self.grid.extraction_area_br
        return (x1 <= ghostCoords[0] <= x2) and (y1 <= ghostCoords[1] <= y2)


    def getAgentObs(self,agentIndex):
        W, H = self.grid.width, self.grid.height
        ax, ay = self.agents[agentIndex].x, self.agents[agentIndex].y
        gx, gy = self.ghostCoords

        # maybe you already include local agent pos + ghost pos
        obs = [
            ax / (W-1), ay / (H-1),
            gx / (W-1), gy / (H-1),
            self.grid.extraction_area_tl[0]/(W-1), self.grid.extraction_area_tl[1]/(H-1),
            self.grid.extraction_area_br[0]/(W-1), self.grid.extraction_area_br[1]/(H-1),

            #Show Progress - USE CHEBYSHEV to match surround logic!
            cheb_dist((ax, ay), (gx, gy)) / max(W-1, H-1),
            self.surroundCounter / numAgents,
            self.holdCounter / self.timeToKill

        ]

        return obs



    #Currently: Full Observation (not partial)
    def full_obs_helper(self):
        """Full-obs helper (normalized coords in [0,1]). Returns list per-agent."""
        W, H = self.grid.width, self.grid.height
        gx, gy = self.ghostCoords
        obs = []
        for i, a in enumerate(self.agents):
            vec = np.array([
                a.x/(W-1), a.y/(H-1),
                gx/(W-1), gy/(W-1),
                self.grid.extraction_area_tl[0]/(W-1), self.grid.extraction_area_tl[1]/(H-1),
                self.grid.extraction_area_br[0]/(W-1), self.grid.extraction_area_br[1]/(H-1),
                
                #Show Progress
                cheb_dist((a.x, a.y), (gx, gy)) / cheb_dist((0,0), (W-1,H-1) ),
                self.surroundCounter / numAgents,
                self.holdCounter / self.timeToKill
            ], dtype=np.float32)
            obs.append(vec)
        return obs

# ----------- Small demo loop -----------
if __name__ == "__main__":

    Game = GameEngine(25,25,29,29, episodeLimit=4000)
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

