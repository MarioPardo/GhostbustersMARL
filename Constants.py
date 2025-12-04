

#Game Constants
TIME_TO_KILL = 5
SURROUND_RADIUS = 3
VISIBILITY_RADIUS = 10


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