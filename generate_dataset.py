from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER
from environment.gym_drone import DroneEnv

import torch
import numpy as np
from torch import nn
import math
from models import *
from core import generate_default_model_name
import wandb

import scipy
import random
from PIL import Image
from random import shuffle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from time import time

from environment.gym_dubins_car import DubinsCarEnv, STEER
from environment.gym_drone import DroneEnv
from environment.gym_dynamic_dubins_multi import MultiDynamicDubinsEnv, STEER

from configs.multi_dynamic_dubins.v0 import *

Env = eval(env_name)
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_'+version_name+'.txt'
BMODEL_PATH = name_dict['db'].replace('.pt', '_'+version_name+'.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_network():
    if ENV_CONFIG['PROB'][1] == 0:
        bnn = eval(MODEL)(HIDDEN_SIZE, keys=['agent'], pos_encode=PE_DIM)
    else:
        bnn = eval(MODEL)(HIDDEN_SIZE, keys=['agent', 'obstacle'], pos_encode=PE_DIM)
    bnn.to(device)
    return bnn


def create_env(num_agents=None, size=None, density=None, simple=None, min_dist=2, **kwargs):
    env_config = deepcopy(ENV_CONFIG)
    if num_agents is not None:
        env_config['num_agents'] = num_agents
    elif VARIABLE_AGENT:
        env_config['num_agents'] = np.random.choice(np.arange(1, 1+ENV_CONFIG['num_agents']))

    if len(env_config['PROB']) > 2:
        density_current = np.random.choice(list(env_config['PROB']))
        env_config['PROB'] = (density_current, density_current)

    if size is not None:
        env_config['SIZE'] = (size, size)
    if density is not None:
        env_config['PROB'] = (density, density)
    if simple is not None:
        env_config['simple'] = simple
    env = Env(**env_config, **kwargs)
    return env


# generate for static
num_agents = 3
for density in [0, 5, 10, 15, 20]:
    
    num_obstacles = []
    valid_dataset = []
    for _ in tqdm(range(100)):
        random.seed(_)
        np.random.seed(_)
        torch.manual_seed(_)        
        
        env = create_env(num_agents=3, 
                         density=density,
                         size=4, 
                         max_dist=6,
                         keep_sample_obs=True,)
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))
        num_obstacles.append(len(env.world.obstacles))
        
    import pickle as pkl

    with open(f'dataset/{project_name}_static_{density}.pkl', 'wb') as f:
        pkl.dump(valid_dataset, f)
        
    print(density, np.mean(num_obstacles))


# generate for dynamic
for num_agents in [1,3]+list(2**np.arange(3, 9)):

    num_obstacles = []
    valid_dataset = []
    for _ in tqdm(range(100)):
        random.seed(_)
        np.random.seed(_)
        torch.manual_seed(_)

        size = int((num_agents*16)**0.5)
        env = create_env(num_agents=num_agents, 
                         size=size,
                         max_dist=6,
                         density=10,)
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))
        num_obstacles.append(len(env.world.obstacles))
        
    num_obstacles.append(len(env.world.obstacles))
    import pickle as pkl

    with open(f'dataset/{project_name}_dynamic_{num_agents}.pkl', 'wb') as f:
        pkl.dump(valid_dataset, f)
    print(num_agents, np.mean(num_obstacles), size**2)
        
        
# generate for mixed

size = 32
for num_agents in [1,3]+list(2**np.arange(3, 10)):
    
    num_obstacles = []
    valid_dataset = []
    for _ in tqdm(range(100)):
        random.seed(_)
        np.random.seed(_)
        torch.manual_seed(_)        
        
        env = create_env(num_agents=num_agents, 
                         size=size,
                         keep_sample_obs=False,
                         density=1,
                         max_dist=6)
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))
        num_obstacles.append(len(env.world.obstacles))
    
    import pickle as pkl

    with open(f'dataset/{project_name}_mixed_{num_agents}.pkl', 'wb') as f:
        pkl.dump(valid_dataset, f)    

    print(num_agents, np.mean(num_obstacles))        