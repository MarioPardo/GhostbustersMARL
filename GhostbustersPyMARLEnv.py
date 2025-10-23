import numpy as np
from Constants import N_ACTIONS
import GameEngine

class GhostbustersPyMARLEnv:
    def __init__(self, n_agents=4, episode_limit=500, width=30, height=30, extraction=(25,29), seed=0):
        self.n_agents = n_agents
        self.episode_limit = episode_limit
        self.gridwidth, self.gridheight = width, height
        self.rng = np.random.default_rng(seed)

        ex, ey = extraction
        self.engine = GameEngine(extractionX=ex, extractionY=ey, episode_limit=episode_limit)
        self._obs_dim = 6                 # [ax,ay,gx,gy,ex,ey] normalized
        self._state_dim = 8 + 3 + 2 + 1   # all agents (8) + ghost (3) + extraction (2) + time (1)

    # -------- required API --------
    def reset(self):
        self.engine.reset(randomized=True)     # use True later; False for fixed debugging
        return self.get_obs()

    def step(self, actions):
        reward, terminated, info = self.engine.step(actions)
        return reward, terminated, info

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, i):
        W, H = self.gridwidth, self.gridheight
        ax, ay = self.engine.agents[i].x, self.engine.agents[i].y
        gx, gy = self.engine.ghostCoords
        ex, ey = self.engine.grid.extraction_point

        return np.array([
            ax/(W-1), ay/(H-1),
            gx/(W-1), gy/(H-1),
            ex/(W-1), ey/(H-1),
        ], dtype=np.float32)

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

        #Ghost + visible flag (full obs baseline → 1.0)
        gx, gy = self.engine.ghostCoords
        s += [gx/(W-1), gy/(H-1), 1.0]

        #Extraction point
        ex, ey = self.engine.grid.extraction_point
        s += [ex/(W-1), ey/(H-1)]

        #Game progress
        s += [self.engine.t / self.episode_limit]

        return np.asarray(s, dtype=np.float32)

    def get_state_size(self):
        return self._state_dim

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, i):
        #TODO -later: mask invalid moves into walls, other agents, ghost, etc.
        return np.ones((N_ACTIONS,), dtype=np.int32)

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
