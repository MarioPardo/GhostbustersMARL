import os, sys
import numpy as np
import pygame


# Ensure your project src/ is on the path so imports work
SOURCE_ROOT = "/home/mariop/Documents/Programming/school/GhostbustersMARL"
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from GameEngine import GameEngine           # your class
from Constants import *


class GhostbustersPyMARLEnv:
    def __init__(self, n_agents, 
                 episode_limit, 
                 width, height, 
                 extraction_tl,
                 extraction_br, 
                 seed=0, **kwargs):
        
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.gridwidth, self.gridheight = width, height
        self.rng = np.random.default_rng(seed)

        e_tl_x, e_tl_y = extraction_tl
        e_br_x, e_br_y = extraction_br

        self.spawn_radius = kwargs.get("spawn_radius", None)
        self.ghost_move_prob = kwargs.get("ghost_move_prob", 1.0)
        self.vision_radius = kwargs.get("vision_radius", None)
        self.ghost_avoid_radius = kwargs.get("ghost_avoid_radius", 2)
        self.surround_radius = kwargs.get("surround_radius", 3)
        
        self._obs_dim = 2 + 3 + (n_agents - 1) * 3 + 4 + 1
        self._state_dim = 2 * n_agents + 3 + 4 + 1 + 3
        
        self.last_reward = 0.0   # all agents (2*n) + ghost (3) + extraction (4) + time (1) + progress (3)

        # Environment parameters from kwargs

        #reward parameters
        self.reward_kill = kwargs.get("reward_kill", 50)
        self.lambda_ghost_dist_to_extraction = kwargs.get("lambda_ghost_dist_to_extraction", 0.05)
        self.lambda_agent_dist_to_ghost = kwargs.get("lambda_agent_dist_to_ghost", 1.0)
        self.reward_surrounded = kwargs.get("reward_surrounded", 5)
        self.lambda_quadrant_coverage = kwargs.get("lambda_quadrant_coverage", 0.7)
        self.reward_new_surround = kwargs.get("reward_new_surround", 5)
        self.reward_is_extracting = kwargs.get("reward_is_extracting", 20)
        self.reward_ghost_spotted = kwargs.get("reward_ghost_spotted", 10)
        self.reward_ghost_visible = kwargs.get("reward_ghost_visible", 0.5)
        self.lambda_agent_spread = kwargs.get("lambda_agent_spread", 0.0)
        self.lambda_grid_coverage = kwargs.get("lambda_grid_coverage", 0.0)

        rewardcfg = {
            "reward_kill": self.reward_kill,
            "lambda_ghost_dist_to_extraction": self.lambda_ghost_dist_to_extraction,
            "lambda_agent_dist_to_ghost": self.lambda_agent_dist_to_ghost,
            "reward_surrounded": self.reward_surrounded,
            "lambda_quadrant_coverage": self.lambda_quadrant_coverage,
            "reward_new_surround": self.reward_new_surround,
            "reward_is_extracting": self.reward_is_extracting,
            "reward_ghost_spotted": self.reward_ghost_spotted,
            "reward_ghost_visible": self.reward_ghost_visible,
            "lambda_agent_spread": self.lambda_agent_spread,
            "lambda_grid_coverage": self.lambda_grid_coverage,

        }

        #rendering
        
        self.renderEnabled = kwargs.get("render", False)              
        self.render_every = int(kwargs.get("render_every", 100))
        self._have_display = False
    
        self.spawn_radius = kwargs.get("spawn_radius", None)
        self.ghost_move_prob = kwargs.get("ghost_move_prob", 1.0)
        self.vision_radius = kwargs.get("vision_radius", None)        #default stuff
        self.map_name = kwargs.pop("map_name", "ghostbusters")
        self.key = kwargs.pop("key", None)
        self.common_reward = kwargs.pop("common_reward", None)
        if kwargs:
            print(f"[GhostbustersEnv] Unused env args ignored: {list(kwargs.keys())}")


        #Create the Game Engine
        self.engine = GameEngine(
            extractionTL_x=e_tl_x,
            extractionTL_y=e_tl_y,
            extractionBR_x=e_br_x,
            extractionBR_y=e_br_y,
            episodeLimit=episode_limit,
            reward_cfg=rewardcfg,
            spawn_radius=self.spawn_radius,  
            ghost_move_prob=self.ghost_move_prob,
            vision_radius=self.vision_radius,
            ghost_avoid_radius=self.ghost_avoid_radius,
            surround_radius=self.surround_radius
        )

    # -------- required API --------
    def reset(self):
        self.engine.reset(randomized=True)     # use True later; False for fixed debugging

        #rendering
        if self.renderEnabled and not self._have_display:
            self.engine.grid.init_display(cell_size=20, caption="Ghostbusters – training view")
            self._have_display = True
        if self.renderEnabled and self._have_display:
            self.engine.grid.draw_grid(show_grid_lines=True)


        return self.get_obs()

    def step(self, actions):
        reward, terminated, info = self.engine.step(actions)
        truncated = self.engine.Time >= self.episode_limit
        obs = self.get_obs()
        
        self.last_reward = reward

        if self.renderEnabled:
            if self._have_display and (self.engine.Time % self.render_every == 0):
                self.render()

        return obs, reward, terminated,truncated, info
    
    def render(self):
        if not self._have_display:
            print("[GhostbustersEnv] Initializing pygame display for rendering...")
            self.engine.grid.init_display(cell_size=18, caption="Ghostbusters")
            self._have_display = True
            print("[GhostbustersEnv] Display initialized successfully")

        self.engine.grid.draw_grid(show_grid_lines=True)

        pygame.event.pump()
        pygame.display.set_caption(f"t={self.engine.Time}  r={self.last_reward:.2f}  vis={int(self.engine.ghost_visible)}")


    def close(self):
        if self._have_display:
            self.engine.grid.close_display()
            self._have_display = False

    def getAllAvailableActions(self):
        return self.engine.getAllAvailableActions()

    def get_obs(self):
        return [self.engine.getAgentObs(i) for i in range(self.n_agents)]


    def get_obs_size(self):
        return self._obs_dim

    def get_state(self):
        """Full state representation (normalized coords in [0,1] plus time).

        
        """

        W, H = self.gridwidth, self.gridheight
        s = []

        # all agents
        for a in self.engine.agents:
            s += [a.x/(W-1), a.y/(H-1)]

        if self.engine.ghost_visible:
            gx, gy = self.engine.ghostCoords
            s += [gx/(W-1), gy/(H-1), 1.0]
        else:
            s += [0.0, 0.0, 0.0]

        #Extraction area
        etlx, etly = self.engine.grid.extraction_area_tl
        ebrx, ebry = self.engine.grid.extraction_area_br
        s += [etlx/(W-1), etly/(H-1), ebrx/(W-1), ebry/(H-1)]

        #Game progress
        s += [self.engine.Time / self.episode_limit]

        #Global task progress (for value estimation)
        s += [
            self.engine.surroundCounter / self.n_agents,
            self.engine.holdCounter / self.engine.timeToKill,
        ]
        
        #Only include avg distance to ghost if visible
        if self.engine.ghost_visible:
            avg_dist = sum(cheb_dist((a.x, a.y), self.engine.ghostCoords) for a in self.engine.agents) / (self.n_agents * max(W-1, H-1))
            s += [avg_dist]
        else:
            s += [0.0]

        return np.asarray(s, dtype=np.float32)

    def get_state_size(self):
        return self._state_dim

    def get_avail_actions(self):
        return [self.engine.getValidAgentActions(i) for i in range(self.n_agents)]


    def get_total_actions(self):
        return N_ACTIONS

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }


