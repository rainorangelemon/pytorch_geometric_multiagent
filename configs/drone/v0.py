version_name = 'v0'

# explore
DECAY_EXPLORE_RATE = 0.9
DECAY_NOMINAL_RATE = 0.
MIN_EXPLORE_EPS = 0.0
MAX_EXPLORE_EPS = 0.5
EXPLORE_WAY = 'exponential'
NOMINAL_WAY = 'exponential'

# training
LR = 3e-4
MIN_LR = 1e-8
PATIENCE = 2
BATCH = 256
USE_SCHEDULER = True
OPTIMIZER = 'Adam'
CLIP_NORM = True
ALL_LIE = False

# algorithm
POTENTIAL_OBS = False
VARIABLE_AGENT = False
CBUF_BEFORE_RELABEL = True
THRESHOLD = 2e-2
SPATIAL_PROP = False
n_candidates = 2000

# dataset
TRAIN_ON_HARD = False
N_DATASET = 10
N_VALID_DATASET = 50
MAX_VISIT_TIME = 1000

# relabel
RELABEL = True
DECAY_RELABEL = False
REFINE_EPS = 1.0
RELABEL_ONLY_AGENT = False
ONLY_BOUNDARY = False
DANGER_THRESHOLD = 0
DYNAMIC_RELABEL = False

# buffer size
N_TRAJ = N_EPOCH = 1000000000
N_BUFFER = 0
N_DYNAMIC_BUFFER = 10000
N_TRAJ_BUFFER = 60000
N_CBUF = 60000

# training speed & validation
UPDATE_FREQ = 4
N_EVALUATE = 400
N_VALID = 400
N_ITER = 50
N_TRAJ_PER_EPOCH = 10
N_WARMUP = 0

# model
MODEL = 'OriginGNNv8'
PE_DIM = None
HIDDEN_SIZE = 128

# environment
ENV_CONFIG = {
    'num_agents': 8,
    'SIZE': (4,4),
    'agent_top_k': 6,
    'obstacle_top_k': 2,
    'PROB': (0.2,1.0),
    'simple': False,
}
FIX_ENV = False

# target network
POLYAK = 0.

# not important
SAVE_GIF = False