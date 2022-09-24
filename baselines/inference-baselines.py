import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tqdm import tqdm
import gc
from copy import deepcopy
import torch
import numpy as np
from torch import nn
import math
from models import *   
from ppo import *
from core import generate_default_model_name
import pickle as pkl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SAVE_GIF = False

for NUM_AGENTS in [3]:#,2,4,8,16,32,64,128,256,512,1024,2048]:

    # with open(f'dataset/{project_name}_{NUM_AGENTS}.pkl', 'rb') as f:
    #     valid_dataset = pkl.load(f)

    bnn = create_network(mode='actor')
    print(bnn.load_state_dict(torch.load(ACTOR_PATH.replace('.pt', '_current.pt'), map_location=device)))
    bnn.eval();

    import copy

    collideds = []
    dones = []
    lengths = []
    unsafe_rates = []
    paths = []
    neighbor_a = []
    neighbor_o = []

    path = f'gifs/0519/{project_name}_{version_name}/{NUM_AGENTS}'
    os.makedirs(path, exist_ok=True)

    for v_idx in tqdm(range(100)):#len(valid_dataset))):
        torch.manual_seed(v_idx)
        random.seed(v_idx)
        np.random.seed(v_idx)
        
        # data = valid_dataset[v_idx]
        env = create_env()#num_agents=NUM_AGENTS, size=max(int((NUM_AGENTS*2)**0.5), 4), max_dist=1, density=0) #
        # env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
        if SAVE_GIF:
            gif_file = f'gifs/0519/{project_name}_{version_name}/{NUM_AGENTS}/'+str(v_idx)+f'.gif'
        else:
            gif_file = None
        collided, done, path, unsafe_rate = infer(env,bnn,verbose=False,stop_at_collision=False,need_gif=gif_file)
        collideds.append(collided)
        dones.append(done)
        lengths.append(len(path))
        unsafe_rates.append(unsafe_rate)
        paths.append(path)
        
    # for agent in path:
    #     env.world.agents = agent.copy()

    print(NUM_AGENTS, np.any(collideds, axis=-1).mean(), np.mean(collideds), np.mean(unsafe_rates), np.mean(dones), np.mean(lengths))

    with open(f'dataset/results/{project_name}_{version_name}_{NUM_AGENTS}.pkl', 'wb') as f:
        pkl.dump({'collideds': collideds,
                  'unsafe_rates': unsafe_rates,
                  'dones': dones,
                  'lengths': lengths,
                  'paths': paths}, f)