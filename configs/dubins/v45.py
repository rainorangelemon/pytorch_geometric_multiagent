version_name = 'v45'
OBSTACLE_DENSITY = 1.0
LR = 3e-4
MIN_LR = 1e-8
PATIENCE = 2
DECAY_EXPLORE_RATE = 0.97
DECAY_NOMINAL_RATE = 0.2
MIN_EXPLORE_EPS = 0.
MAX_EXPLORE_EPS = 0.5
POTENTIAL_OBS = True
PREFERENCE_OBS = True
TRAIN_ON_HARD = False
VARIABLE_AGENT = False
CBUF_BEFORE_RELABEL = False
PER_STEP_EXPLORE = True
CBUF_ONLY_BOUNDARY = False
OUTDATE = float('inf')


N_TRAJ = N_EPOCH = 1000000
N_CBUF = 0

NUM_AGENTS = 8
MAP_SIZE = 4

n_candidates = 2000
BATCH = 64
N_ITER = 50
N_TRAJ_PER_EPOCH = 100
N_BUFFER = 200
N_EVALUATE = 100
N_VALID = 100
N_WARMUP = 100
N_DATASET = 10
N_VALID_DATASET = 20
THRESHOLD = 1.5e-2
HIDDEN_SIZE = 128
RELABEL = True
CYCLIC = False
DECAY_RELABEL = False
USE_SCHEDULER = False
OPTIMIZER = 'Adam'
SAVE_GIF = False
MAX_GRAD_NORM = 0.5