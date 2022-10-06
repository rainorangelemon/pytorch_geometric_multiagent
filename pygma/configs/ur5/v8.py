project_name = 'ur5'
env_name = 'UR5Env'
version_name = 'v8'

# explore
DECAY_EXPLORE_RATE = 1.0
DECAY_NOMINAL_RATE = 0.
MIN_EXPLORE_EPS = 0.1
MAX_EXPLORE_EPS = 0.1
EXPLORE_WAY = 'exponential'
NOMINAL_WAY = 'exponential'
SAFE_EXPLORE = False
ALL_EXPLORE = False
DANGER_EXPLORE = True

# training
LR = 1e-3
MIN_LR = 1e-5
PATIENCE = 5
BATCH = 256
USE_SCHEDULER = True
OPTIMIZER = 'Adam'
CLIP_NORM = True
ALL_LIE = False
TRAIN_TIMING = 'buffer_size'
KEY_BUFFER = '[cbuf_obstacle]'

# training freq
N_ITER = 100
N_TRAJ_PER_UPDATE = 12
UPDATE_FREQ = 4
SHARE_SAMPLE_ACROSS_UPDATE = True

# algorithm
POTENTIAL_OBS = False
VARIABLE_AGENT = False
CBUF_BEFORE_RELABEL = True
THRESHOLD = 0
GAMMA = 0
DISCOUNT = 1
LIE_DERIVE_SAFE = False
SPATIAL_PROP = False
n_candidates = 2000

# dataset
TRAIN_ON_HARD = False
N_DATASET = 1
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
N_DYNAMIC_BUFFER_FREE = 6000
N_DYNAMIC_BUFFER_DANGER = 6000
N_TRAJ_BUFFER = 200000
N_CBUF_AGENT = 100000
N_CBUF_OBSTACLE = 100000

# training speed & validation
N_EVALUATE = 400
N_VALID = 400
N_WARMUP = 0

# model
MODEL = 'OriginGNNv11'
PE_DIM = None
HIDDEN_SIZE = 128

# environment
ENV_CONFIG = {
    'hetero': True,
    'num_agents': 2,
    'SIZE': (3,3),
    'agent_top_k': 2,
    'obstacle_top_k': 2,
    'agent_obs_radius': 1.5,
    'obstacle_obs_radius': 1.5,    
    'PROB': (0.,30),
    'angle_embed': True,
    'simple': False,
    'min_dist': 2,
}
OBS_CONFIG = {
    'share_weight': True,
    'rgraph_a': True, 
    'rgraph_o': True,
    'has_goal': False, 
}

OBS_CONFIG_DECOMPOSE = {
    'share_weight': True,
    'rgraph_a': True, 
    'rgraph_o': True,
    'n_sub_o': (1,9),
    'n_sub_a': (0,2),
    'has_goal': False, 
    'iteration': 1,
}

FIX_ENV = False

# target network
POLYAK = 0.

# not important
SAVE_GIF = False