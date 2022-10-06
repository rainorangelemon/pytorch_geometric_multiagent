from tqdm import tqdm
import gc
from copy import deepcopy
import torch
import numpy as np
from torch import nn
import math
from models import *   
from arm_env2 import *
from core import generate_default_model_name
import pickle as pkl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SAVE_GIF = True

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

path = f'gifs/0530/{version_name}'
os.makedirs(path, exist_ok=True)

arm_ids = [0,1,2]
ids_str = ''
for id_ in arm_ids:
    ids_str = ids_str + str(id_)
if SAVE_GIF:
    gif_file = f'gifs/0530/{version_name}/'+ids_str+'.mp4'
else:
    gif_file = None
env = create_env(num_agents=len(arm_ids), arm_ids=arm_ids, randomize=False)
print(env.world.arm_ids)
# print(env.world.get_status())
# assert False
collided, done, path = infer(env,bnn,verbose=True,n_action=20000, 
                             spatial_prop=False,
                             lie_derive_safe=False,
                             seed=0,
                             decompose='random',
                             stop_at_collision=False,need_gif=gif_file)
collideds.append(collided)
dones.append(done)
lengths.append(len(path))
paths.append(path)

# for agent in path:
#     env.world.agents = agent.copy()

print(np.any(collideds, axis=-1).mean(), np.mean(collideds), np.mean(dones), np.mean(lengths))

# with open(f'dataset/results/{project_name}.pkl', 'wb') as f:
#     pkl.dump({'collideds': collideds,
#               'dones': dones,
#               'lengths': lengths,
#               'paths': paths}, f)
