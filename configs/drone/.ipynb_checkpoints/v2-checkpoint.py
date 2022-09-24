project_name = 'drone'
env_name = 'DroneEnv'
version_name = 'v2'

# explore
DECAY_EXPLORE_RATE = 1.0
DECAY_NOMINAL_RATE = 0.
MIN_EXPLORE_EPS = 0.1
MAX_EXPLORE_EPS = 0.5
ALL_EXPLORE = False
EXPLORE_WAY = 'exponential'
NOMINAL_WAY = 'exponential'

# training
LR = 1e-3
MIN_LR = 1e-5
PATIENCE = 3
BATCH = 256
USE_SCHEDULER = True
OPTIMIZER = 'Adam'
CLIP_NORM = True
ALL_LIE = False
UPDATE_FREQ = 4

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
N_EVALUATE = 800
N_VALID = 800
N_TRAJ_PER_EPOCH = 20
N_ITER = 50
N_WARMUP = 0

# model
MODEL = 'OriginGNNv10'
PE_DIM = None
HIDDEN_SIZE = 128

# environment
ENV_CONFIG = {
    'num_agents': 3,
    'SIZE': (3,3),
    'agent_top_k': 2,
    'obstacle_top_k': 1,
    'PROB': (0.2,1.0),
    'angle_embed': True,    
    'simple': False,
}
FIX_ENV = False

# target network
POLYAK = 0.

# not important
SAVE_GIF = False