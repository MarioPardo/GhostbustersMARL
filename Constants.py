

#Game Constants
TIME_TO_KILL = 10
SURROUND_RADIUS = 2


#Actions
ACTION_TO_DELTA = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0), (0,  0), (1,  0),
    (-1,  1), (0,  1), (1,  1),
]
N_ACTIONS = len(ACTION_TO_DELTA)



### FUNCTIONS ###
def cheb_dist(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))