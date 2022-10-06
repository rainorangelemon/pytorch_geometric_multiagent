project_name = 'chenning_ppo'
env_name = 'DubinsCarEnv'
version_name = 'ppo_v0'

# ddpg parameter
CLIP_RATIO = 0.2
N_WARMUP = 100
SHARE_SAMPLE_ACROSS_UPDATE = False
GAMMA = 0.95
LAMBDA = 0.95
POLYAK = 0.99
TARGET_KL = 0.01
N_BUFFER = 2048
BATCH = 2048
N_ITER = 40
MAX_CLIP_NORM = 0.5

MAX_ACTION_STD = 0.4 
ACTION_STD_DECAY_RATE = 0
MIN_ACTION_STD = 0.4
ACTION_STD_DECAY_FREQ = 10

# explore
DECAY_EXPLORE_RATE = 0.9
DECAY_NOMINAL_RATE = 0.9
MIN_EXPLORE_EPS = 0.1
MAX_EXPLORE_EPS = 1.0
EXPLORE_WAY = 'exponential'
NOMINAL_WAY = 'exponential'
SAFE_EXPLORE = True
ALL_EXPLORE = False
DANGER_EXPLORE = True

# training
PI_LR = 3e-4
V_LR = 1e-3
MIN_LR = 1e-5
PATIENCE = 3
USE_SCHEDULER = False
OPTIMIZER = 'Adam'
CLIP_NORM = True
ALL_LIE = False

# training freq
N_TRAJ_PER_UPDATE = 10
UPDATE_FREQ = 4

# algorithm
POTENTIAL_OBS = False
VARIABLE_AGENT = False
CBUF_BEFORE_RELABEL = True
THRESHOLD = 2e-2
LIE_DERIVE_SAFE = False
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
DANGER_THRESHOLD = 2e-2
DYNAMIC_RELABEL = False

# buffer size
N_TRAJ = N_EPOCH = 1000000000
N_DYNAMIC_BUFFER = 3000
N_TRAJ_BUFFER = 1e6
N_CBUF = 10000

# training speed & validation
N_EVALUATE = 400
N_VALID = 400

# model
MODEL = 'OriginGNNv11'
PE_DIM = None
HIDDEN_SIZE = 128

# environment
ENV_CONFIG = {
    'hetero': True,
    'num_agents': 3,
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

FIX_ENV = True

# not important
SAVE_GIF = False