

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

    def __init__(self, extractionTL_x: int, extractionTL_y: int, extractionBR_x:int, extractionBR_y, episodeLimit: int, reward_cfg, spawn_radius: int, ghost_move_prob: float, vision_radius: int = 6, ghost_avoid_radius: int = 2, surround_radius: int = 3):
        
        self.spawn_radius = spawn_radius
        self.vision_radius = vision_radius if vision_radius is not None else 15
        self.ghost_avoid_radius = ghost_avoid_radius
        self.surround_radius = surround_radius
        self.grid = Grid(extractionTL_x, extractionTL_y, extractionBR_x, extractionBR_y, visibilityRadius=self.vision_radius)
        self.agentCoords, self.ghostCoords = self.SpawnEntities(randomized=True, spawn_radius=spawn_radius)
        self.grid.setEntities(self.agentCoords, self.ghostCoords)
        
        self.ghost_visible = False
        self.ghost_last_seen_pos = None 

        #Create Agents, Ghost
        self.agents = []
        for i, (ax, ay) in enumerate(self.agentCoords):
            agent = Agent(ax, ay, agent_id=i)
            self.agents.append(agent)

        self.ghost = Ghost(self.ghostCoords[0], self.ghostCoords[1], movementProb=ghost_move_prob, avoidRadius=self.ghost_avoid_radius, surroundRadius=self.surround_radius)


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
        self.reward_is_extracting = reward_cfg.get("reward_is_extracting", 20)
        self.reward_ghost_spotted = reward_cfg.get("reward_ghost_spotted", 10)
        self.reward_ghost_visible = reward_cfg.get("reward_ghost_visible", 0.5)
        self.lambda_agent_spread = reward_cfg.get("lambda_agent_spread", 0.0)
        self.lambda_grid_coverage = reward_cfg.get("lambda_grid_coverage", 0.0)

        self.timeToKill = TIME_TO_KILL
        
        # Initialize visit map for exploration tracking
        self.visit_map = {}
        self.prev_agentCoords = None  # Track previous positions for movement detection

        self.Time = 0
        self.holdCounter = 0
        self.surroundCounter = 0
        self.prev_ghost_visible = False
        self.time_first_seen = None

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
        self.ghost_visible = False
        self.prev_ghost_visible = False
        self.time_first_seen = None
        self.reward_ghost_spotted = self.reward_cfg.get("reward_ghost_spotted", 10)
        self.visit_map = {}  # Reset exploration tracking
        self.prev_agentCoords = None  # Reset movement tracking
        self.updateGhostVisibility()

        return self.full_obs_helper()


    def SpawnEntities(self, randomized: bool, spawn_radius: int):
        """
        Spawn agents and ghost at random non-overlapping locations.
        
        Args:
            randomized: If False, use preset positions
            spawn_radius: If provided, spawn ghost within this radius of at least one agent (curriculum)
        """

        if not randomized:
            return presetAgentCoords, presetGhostCoords

        
        # First spawn agents 
        agentSpawnpoints = []
        for i in range(numAgents):
            while True:
                ax = np.random.randint(0, self.grid.width)
                ay = np.random.randint(0, self.grid.height)
                
                if (ax, ay) not in agentSpawnpoints:
                    agentSpawnpoints.append((ax, ay))
                    break
        
        #Ghost spawns atleast some distance away from everyone
        attempts = 0
        spawn_radius = self.spawn_radius
        while True:
            attempts += 1
            gx = np.random.randint(0, self.grid.width)
            gy = np.random.randint(0, self.grid.height)
            
            if (gx, gy) not in agentSpawnpoints and not self.isExtracting((gx, gy)):
                # Check distance from all agents
                all_far_enough = all(
                    cheb_dist((gx, gy), agent_pos) > spawn_radius
                    for agent_pos in agentSpawnpoints
                )
                if all_far_enough:
                    ghostSpawnpoint = (gx, gy)
                    break

                if attempts > 100:
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
        prev_d_ghost_extract = cheb_dist(self.grid.extraction_point_center, self.ghostCoords)
        prev_d_ghost_agents = [cheb_dist((a.x, a.y), (self.ghost.x, self.ghost.y)) for a in self.agents]
        prev_surround_count = self.ghost.GetSurroundedCount(self.agentCoords)
        prev_agentCoords = self.prev_agentCoords  # Store for movement detection
        

        #Move Ghost FIRST 
        self.ghostCoords = self.ghost.move(self.agentCoords)

        #Apply agent moves
        for a, act_id in zip(self.agents, actions):
            a.apply_action(act_id)
        self.agentCoords = [(a.x, a.y) for a in self.agents]
        self.prev_agentCoords = self.agentCoords # Store for next step

        prev_ghost_visible = self.ghost_visible
        self.updateGhostVisibility()
       
        self.grid.setEntities(self.agentCoords, self.ghostCoords)
        self.surroundCounter = self.ghost.GetSurroundedCount(self.agentCoords)
        self.grid.setSurrendered(self.surroundCounter >= numAgents)

        #reward & termination
        self.Time += 1
        reward, terminated, sucess = self.globalReward(prev_d_ghost_extract, prev_d_ghost_agents, prev_surround_count, prev_ghost_visible, prev_agentCoords)

        info = {
            "t": self.Time, 
            "hold": self.holdCounter, 
            "success": sucess,
            "time_first_seen": self.time_first_seen if self.time_first_seen is not None else self.Time 
        }
        return reward, terminated, info

    def globalReward(self, prev_d_ghost_extract, prev_d_ghost_agents, prev_surround_count, prev_ghost_visible, prev_agentCoords):
        terminated = False
        success = False
        reward = -0.01

        self.calculateUsefulMetrics

        if self.surroundCounter >= numAgents:
            self.holdCounter += 1
        else:
            self.holdCounter = 0

        # End conditions
        if self.holdCounter >= self.timeToKill : 
            reward += self.win_reward 
            terminated = True
            success = True
        elif self.Time >= self.episodeLimit:
            terminated = True

        if terminated:
            return reward, terminated, success


        ####First time finding ghost
        if not prev_ghost_visible and self.ghost_visible and (self.time_first_seen is None) :
            reward += self.reward_ghost_spotted
            self.reward_ghost_spotted *= 0.8  # Decay for future sightings


        if not self.ghost_visible:
            # Only reward exploration if agents are actually moving
            if prev_agentCoords is not None:
                num_agents_moved = sum(1 for i, (x, y) in enumerate(self.agentCoords) if (x, y) != prev_agentCoords[i])
                if num_agents_moved > 0:
                    # Agents moving - give exploration rewards
                    reward += self.GridCoverageReward()
                    reward += self.AgentSpreadReward()
                else:
                    # All agents standing still: penalty
                    reward -= 0.5
            else:
                # First step of episode:  give rewards
                reward += self.GridCoverageReward()
                reward += self.AgentSpreadReward()

        ####Catching Ghost
        if self.ghost_visible:
            reward += 2* self.AgentToGhostDist_Reward(prev_d_ghost_agents)
            reward += self.Surround_Reward(prev_surround_count)
            reward += self.QuadrantCoverage_Reward()  

        return reward, terminated, success

    def AgentSpreadReward(self):
        avg_dist = 0.0
        count = 0
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                dist = cheb_dist((self.agents[i].x, self.agents[i].y), (self.agents[j].x, self.agents[j].y))
                avg_dist += dist
                count += 1
        if count > 0:
            avg_dist /= count
        max_dist = max(self.grid.width, self.grid.height) - 1
        return self.lambda_agent_spread * (avg_dist / max_dist)
    
    def GridCoverageReward(self):
        exploration_reward = 0.0
        
        for agent in self.agents:
            cell = (agent.x, agent.y)
            visit_count = self.visit_map.get(cell, 0)
            
            # Reward inversely proportional to visit count
            # First visit = full reward, subsequent visits = diminishing returns
            cell_reward = 1.0 / (1.0 + visit_count)
            exploration_reward += self.lambda_grid_coverage * cell_reward
            
            # Update visit count
            self.visit_map[cell] = visit_count + 1
        
        return exploration_reward

    def AgentClumpingPenalty(self):
        penalty = 0.0
        CLUMP_DISTANCE = 2
        
        # Only penalize if not currently surrounding ghost
        if self.surroundCounter >= numAgents:
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

        # Closer to ghost = higher reward, but not too strong to avoid mindless chasing
        max_dist = max(self.grid.width - 1, self.grid.height - 1)
        
        for a in self.agents:
            dist = cheb_dist((a.x, a.y), self.ghostCoords)
            norm_dist = dist / max_dist
            reward += self.lambda_agent_dist_to_ghost * ((1.0 - norm_dist) ** 3) * 3

        return reward
    
    def Surround_Reward(self, prev_surround_count):
        reward = 0.0
        new_surround_count = self.ghost.GetSurroundedCount(self.agentCoords)

        # Progressive surround rewards : bonus for entering radius
        if new_surround_count > prev_surround_count:
            reward += self.reward_new_surround * (new_surround_count - prev_surround_count) 
        
        # PENALTY: Breaking formation (agents leaving surround)
        if new_surround_count < prev_surround_count:
            penalty = self.penalty_formation_break * (prev_surround_count - new_surround_count) * 0.5
            reward -= penalty
        
        
        # Keeping full surround
        if new_surround_count >= numAgents:
            reward += self.reward_full_surround

        return reward

    def QuadrantCoverage_Reward(self):
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
                normalized_dist = direction_distances[i] / max_dist
                proximity_multiplier = (2.0 - normalized_dist) ** 2  
                
                total_reward += self.lambda_quadrant_coverage * proximity_multiplier
        
        return total_reward
    
    def DirectionalPush_Reward(self):
        #Reward agents for positioning to PUSH the ghost toward extraction.

        gx, gy = self.ghostCoords
        ex, ey = self.grid.extraction_point_center
        
        # Vector from ghost to extraction (desired direction for ghost to move)
        ghost_to_extract_x = ex - gx
        ghost_to_extract_y = ey - gy
        extract_dist = max(abs(ghost_to_extract_x), abs(ghost_to_extract_y))  # Chebyshev distance
        
        # Avoid division by zero if ghost is at extraction
        if extract_dist < 0.1:
            return 0.0
        
        # Normalize the extraction direction vector
        extract_dir_x = ghost_to_extract_x / extract_dist
        extract_dir_y = ghost_to_extract_y / extract_dist
        
        total_reward = 0.0
        max_dist = max(self.grid.width, self.grid.height) - 1
        
        for ax, ay in self.agentCoords:
            ghost_to_agent_x = ax - gx
            ghost_to_agent_y = ay - gy
            agent_dist = cheb_dist((ax, ay), (gx, gy))
            
            
            # Normalize agent direction vector
            agent_dir_x = ghost_to_agent_x / max(agent_dist, 1)
            agent_dir_y = ghost_to_agent_y / max(agent_dist, 1)
            
            # Dot product: want agents opposite extraction direction, so we want negative
            dot_product = (agent_dir_x * extract_dir_x + agent_dir_y * extract_dir_y)
            
            # Calculate proximity bonus. Reward when close
            normalized_dist = agent_dist / max_dist
            proximity_multiplier = max(0, (1.0 - normalized_dist)) ** 1.5 
            
            # Simple exponential reward: maximize alignment opposite to extraction direction
            alignment_score = (-dot_product + 1) / 2  # Maps [-1, 1] to [1.0, 0.0]
            directional_reward = self.lambda_quadrant_coverage * proximity_multiplier * (alignment_score ** 3)
            total_reward += directional_reward  

        return total_reward

    def GhostToExtraction_Reward(self, prev_d_ghost_extract):
        reward = 0.0

        max_dist = max(self.grid.width - 1, self.grid.height - 1)
        current_dist = cheb_dist(self.grid.extraction_point_center, self.ghostCoords)
        norm_dist = current_dist / max_dist
        
        # Reward for being close to extraction (static distance reward)
        reward += self.lambda_ghost_dist_to_extraction * ((1.0 - norm_dist) ** 3) * 5.0

        distance_improvement = prev_d_ghost_extract - current_dist
        if distance_improvement > 0:
            reward += distance_improvement * self.lambda_ghost_dist_to_extraction * 20.0
        elif distance_improvement < 0:
            reward += distance_improvement * self.lambda_ghost_dist_to_extraction * 10.0

        # Ghost enters extraction zone
        is_extracting = self.isExtracting(self.ghostCoords)
        if is_extracting:
            reward +=  self.reward_is_extracting  #
        
        #Hold in extraction zone
        if is_extracting and self.surroundCounter >= numAgents:
            reward += 10.0 * (self.holdCounter / self.timeToKill)  

        return reward




    def calculateUsefulMetrics(self):
        
        # Average Distance Agents to Ghost
        total_dist = sum(cheb_dist((a.x, a.y), self.ghostCoords) for a in self.agents)
        self.avgDistToGhost = total_dist / len(self.agents)
        

        return

    def isExtracting(self, ghostCoords):
        (x1, y1) = self.grid.extraction_area_tl
        (x2, y2) = self.grid.extraction_area_br
        return (x1 <= ghostCoords[0] <= x2) and (y1 <= ghostCoords[1] <= y2)

    def canSeeGhost(self, agent_index):
        if self.vision_radius is None:
            return False
        ax, ay = self.agents[agent_index].x, self.agents[agent_index].y
        gx, gy = self.ghostCoords
        return cheb_dist((ax, ay), (gx, gy)) <= self.vision_radius

    def canSeeAgent(self, observer_index, target_index):
        if self.vision_radius is None:
            return True
        if observer_index == target_index:
            return True
        ax, ay = self.agents[observer_index].x, self.agents[observer_index].y
        tx, ty = self.agents[target_index].x, self.agents[target_index].y
        return cheb_dist((ax, ay), (tx, ty)) <= self.vision_radius

    def updateGhostVisibility(self):
        # Debug: check each agent's visibility
        can_see = [self.canSeeGhost(i) for i in range(len(self.agents))]
        self.ghost_visible = any(can_see)
        
        if self.ghost_visible:
            self.ghost_last_seen_pos = self.ghostCoords

            if self.time_first_seen is None:
                self.time_first_seen = self.Time

    def getAgentObs(self, agentIndex):
        W, H = self.grid.width, self.grid.height
        ax, ay = self.agents[agentIndex].x, self.agents[agentIndex].y
        
        obs = [ax / (W-1), ay / (H-1)]
        
        if self.ghost_visible:
            gx, gy = self.ghostCoords
            obs += [gx / (W-1), gy / (H-1), 1.0]
            dist_to_ghost = cheb_dist((ax, ay), (gx, gy)) / max(W-1, H-1)
        else:
            obs += [0.0, 0.0, 0.0]
            dist_to_ghost = 0.0
        
        visible_agents = []
        for other_idx in range(len(self.agents)):
            if other_idx == agentIndex:
                continue
            if self.canSeeAgent(agentIndex, other_idx):
                ox, oy = self.agents[other_idx].x, self.agents[other_idx].y
                dist = cheb_dist((ax, ay), (ox, oy))
                visible_agents.append((dist, ox, oy))
        
        visible_agents.sort(key=lambda x: x[0]) #Sort agent visibility by closest first
        
        for i in range(len(self.agents) - 1):
            if i < len(visible_agents):
                dist, ox, oy = visible_agents[i]
                obs += [ox / (W-1), oy / (H-1), 1.0]
            else:
                obs += [0.0, 0.0, 0.0]
        
        obs += [
            self.grid.extraction_area_tl[0]/(W-1), 
            self.grid.extraction_area_tl[1]/(H-1),
            self.grid.extraction_area_br[0]/(W-1), 
            self.grid.extraction_area_br[1]/(H-1),
            dist_to_ghost
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

    Game = GameEngine(25,25,29,29, episodeLimit=4000, reward_cfg={}, spawn_radius=10,ghost_move_prob= 1)
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

