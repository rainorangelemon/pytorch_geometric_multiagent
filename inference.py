from tqdm import tqdm
import gc
from copy import deepcopy
import torch
import numpy as np
from torch import nn
import math
from models import *   
from v109 import *
from core import generate_default_model_name
import pickle as pkl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SAVE_GIF = False
ONLY_SHOW_COLLIDE = True
VERBOSE = True

for NUM_AGENTS in [1,2,4]:#,8,16,32,64,128,256,512,1024,2048]:

    with open(f'dataset/{project_name}_{NUM_AGENTS}.pkl', 'rb') as f:
        valid_dataset = pkl.load(f)

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

    path = f'gifs/0512/{project_name}_{version_name}/{NUM_AGENTS}'
    os.makedirs(path, exist_ok=True)

    for v_idx in tqdm(range(len(valid_dataset))):
        data = valid_dataset[v_idx]
        env = create_env(num_agents=NUM_AGENTS, size=max(int((NUM_AGENTS*2)**0.5), 4), max_dist=1, density=0)
        env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
        if SAVE_GIF:
            gif_file = f'gifs/0512/{project_name}_{version_name}/{NUM_AGENTS}/'+str(v_idx)+f'_decompose_lie.gif'
        else:
            gif_file = None
        collided, done, path = infer(env,bnn,verbose=VERBOSE,n_action=2000,
                                     spatial_prop=False,lie_derive_safe=False,decompose='random',
                                     stop_at_collision=False,need_gif=gif_file,only_show_collide=ONLY_SHOW_COLLIDE)
        collideds.append(collided)
        dones.append(done)
        lengths.append(len(path))
        paths.append(path)
        
    # for agent in path:
    #     env.world.agents = agent.copy()

    print(NUM_AGENTS, np.any(collideds, axis=-1).mean(), np.mean(collideds), np.mean(dones), np.mean(lengths))

    with open(f'dataset/results/{project_name}_{version_name}_{NUM_AGENTS}.pkl', 'wb') as f:
        pkl.dump({'collideds': collideds,
                  'dones': dones,
                  'lengths': lengths,
                  'paths': paths}, f)