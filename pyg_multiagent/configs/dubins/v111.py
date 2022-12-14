version_name = 'v111'

ENV_CONFIG = {
    'num_agents': 8,
    'SIZE': (4,4),
    'agent_top_k': 6,
    'obstacle_top_k': 2,
    'PROB': (0.2,1.0),
    'simple': False,
}

FIX_ENV = False

LR = 3e-4
PATIENCE = 2
DECAY_EXPLORE_RATE = 0.9
DECAY_NOMINAL_RATE = 0.
MIN_EXPLORE_EPS = 0.01
MAX_EXPLORE_EPS = 1.0
POTENTIAL_OBS = False
TRAIN_ON_HARD = True
VARIABLE_AGENT = False
CBUF_BEFORE_RELABEL = True
REFINE_EPS = 1.0
RELABEL_ONLY_AGENT = False
ALL_LIE = False
ONLY_BOUNDARY = False
DANGER_THRESHOLD = 0
DYNAMIC_RELABEL = False
CLIP_NORM = True

MODEL = 'OriginGNNv8'


PE_DIM = None
N_TRAJ = N_EPOCH = 1000000000
N_BUFFER = 1000000000
N_DYNAMIC_BUFFER = 3000
N_TRAJ_BUFFER = 60000
N_CBUF = 60000
MAX_VISIT_TIME = 1000

POLYAK = 0.
SPATIAL_PROP = False
n_candidates = 2000
BATCH = 64
N_ITER = 50
N_TRAJ_PER_EPOCH = 10
N_EVALUATE = 100
N_VALID = 100
N_WARMUP = 0
N_DATASET = 10
N_VALID_DATASET = 50
THRESHOLD = 1e-2
HIDDEN_SIZE = 128
RELABEL = True
EXPLORE_WAY = 'exponential'
NOMINAL_WAY = 'exponential'
DECAY_RELABEL = False
USE_SCHEDULER = True
OPTIMIZER = 'Adam'
SAVE_GIF = False