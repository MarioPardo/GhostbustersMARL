from envs import REGISTRY

env = REGISTRY["ghostbusters"](
    n_agents=4, episode_limit=50, width=30, height=30, extraction=(25,29), seed=0
)

info = env.get_env_info()
print("ENV INFO:", info)

obs = env.reset()
print("obs[0].shape:", obs[0].shape)

import numpy as np
import random
for _ in range(5):
    acts = [random.randrange(env.get_total_actions()) for _ in range(info["n_agents"])]
    r, term, inf = env.step(acts)
    print("step r:", r, "term:", term, "t:", inf.get("t"))
    if term: break
