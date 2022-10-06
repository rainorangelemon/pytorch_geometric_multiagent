from tqdm import tqdm
import gc
from copy import deepcopy
import torch
import numpy as np
from torch import nn
import math
from models import *   
from v0_multi_dynamic_dubins import *
from core import generate_default_model_name
import pickle as pkl
import os
from potential_field import infer_p
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SAVE_GIF = True
ONLY_SHOW_COLLIDE = False
VERBOSE = True
decompose = 'random'
NUM_AGENTS = 3


bnn = create_network()
print(bnn.load_state_dict(torch.load(BMODEL_PATH, map_location=device)))
bnn.eval();

import copy

collideds = []
dones = []
lengths = []
paths = []
neighbor_a = []
neighbor_o = []

path = f'gifs/0820/{project_name}_{version_name}/{NUM_AGENTS}'
os.makedirs(path, exist_ok=True)

env = create_env(num_agents=NUM_AGENTS, size=max(int((NUM_AGENTS*2)**0.5), 2), density=30, max_dist=6)
if SAVE_GIF:
    gif_file = f'gifs/0820/{project_name}_{version_name}/{NUM_AGENTS}.mp4'
else:
    gif_file = None
collided, done, path = infer(env,bnn=bnn,verbose=VERBOSE,n_action=10000,
                             spatial_prop=False,lie_derive_safe=False,decompose=decompose,
                             stop_at_collision=False, stop_at_done=False,
                             max_episode_length=512,
                             need_gif=gif_file,only_show_collide=ONLY_SHOW_COLLIDE)
collideds.append(collided)
dones.append(done)
lengths.append(len(path))
paths.append(path)