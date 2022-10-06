import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER
from environment.gym_drone import DroneEnv
from environment.gym_ur5 import UR5Env
from environment.gym_dynamic_dubins_multi import MultiDynamicDubinsEnv, STEER
import torch.nn.functional as F
from torch_scatter import scatter

import torch
import numpy as np
from torch import nn
import math
from models import *
from core import generate_default_model_name
from torch.distributions.normal import Normal
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

if __name__ == '__main__':
    from baselines.macbf_config_multi_dynamic_dubins import *
else:
    from baselines.macbf_config_multi_dynamic_dubins import *

# torch.autograd.set_detect_anomaly(True)

POLYAK = 0.995

Env = eval(env_name)
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_'+version_name+'.txt'
BMODEL_PATH = ACTOR_PATH = name_dict['db'].replace('.pt', '_'+version_name+'_actor.pt')
CRITIC_PATH = name_dict['db'].replace('.pt', '_'+version_name+'_critic.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_env(num_agents=None, size=None, density=None, simple=None, min_dist=2, **kwargs):
    if FIX_ENV:
        env = Env(num_agents=3, SIZE=(3,3), agent_top_k=2, obstacle_top_k=1, PROB=(0.,1.0), simple=False,)
        env.world.obstacles = np.array([[1.5, 1.5]])
        env.world.agents = np.array([[0.5, 0.5, 0],
                                     [2.5, 2.5, 0],
                                     [0.5, 2.5, 0]])
        env.world.agent_goals = np.array([[2.5, 2.5],
                                          [0.5, 0.5],
                                          [2.5, 0.5]])
    else:
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


def create_network(mode='actor'):
    env = create_env()
    if ENV_CONFIG['PROB'][1] == 0:
        nn = RLNet(HIDDEN_SIZE, 1 if mode=='critic' else env.action_dim, 
                   use_tanh=False, keys=['agent', 'goal'], pos_encode=PE_DIM)
    else:
        nn = RLNet(HIDDEN_SIZE, 1 if mode=='critic' else env.action_dim, 
                   use_tanh=False, keys=['agent', 'obstacle', 'goal'], pos_encode=PE_DIM)
    nn.to(device)
    return nn


def train_barrier(critic, actor, optimizer, buf_f, buf_d, env, n_iter):
    
    def compute_loss_a(data):
        action = actor(data)
        return (((action-data['oracle_action'])**2).sum(dim=-1)*data['prev_free']).sum() / (1e-9 + data['prev_free'].sum())

    def compute_loss_cbf(data):
        value = critic(data).reshape(-1)

        bloss_f = ((1e-2-value).relu()*data['prev_free']) / (1e-9 + data['prev_free'].sum())
                #  / (1e-9 + data['prev_free'].sum())        
        bloss_d = ((1e-2+value).relu()*data['prev_danger']) / (1e-9 + data['prev_danger'].sum())
        
        # max_dist = torch.zeros(len(data['agent'].x)).to(device)
        # min_dist = 10*torch.ones(len(data['agent'].x)).to(device)
        # scatter(data['o_near_a'].edge_attr[:, 5:7].norm(dim=-1), data['o_near_a'].edge_index[1, :], dim=-1, reduce="max", out=max_dist)
        # scatter(data['o_near_a'].edge_attr[:, 5:7].norm(dim=-1), data['o_near_a'].edge_index[1, :], dim=-1, reduce="min", out=min_dist)

        next_data = data.clone()
        data = data.clone()
        data['action'] = actor(data)
        next_pos = env.dynamic_torch(data['agent'].pos, data['action'])
        next_data = env.obs_from_pos(next_data, next_pos, **OBS_CONFIG)
        next_value = critic(next_data).reshape(-1)
        deriv = (next_value-value+0.1*value)
        dloss = ((-deriv+1e-2).relu())
        # try:
        #     return bloss_f.sum(), bloss_d.sum(), dloss.mean(), (min_dist[data['prev_danger'].bool()].min(), min_dist[data['prev_danger'].bool()].max()), (min_dist[data['prev_free'].bool()].min(), max_dist[data['prev_free'].bool()].max())
        # except:
        return bloss_f.sum(), bloss_d.sum(), dloss.mean(), (0,0), (10,10)
            

    for _ in range(n_iter):
        for buf in [buf_f, buf_d]:
            
            if len(buf.obs_buf)==0:
                continue
            data = buf.next_data().clone()
            data = data.to(device)  
            optimizer.zero_grad()
            loss_a = compute_loss_a(data)
            bloss_f, bloss_d, dloss, o1, o2 = compute_loss_cbf(data)
            if buf is buf_f:
                (1e-1 * loss_a + bloss_f + bloss_d + dloss).backward()
            elif buf is buf_d:
                (bloss_f + bloss_d + dloss).backward()
            optimizer.step()

            wandb.log({"loss/prev_free": data['prev_free'].sum(), 
                       "loss/prev_danger": data['prev_danger'].sum(), 
                       "loss/bloss_f": bloss_f,
                       "loss/bloss_d": bloss_d,
                       "loss/dloss": dloss,
                       "loss/aloss": loss_a,})
                       # "loss/danger_dist_min": o1[0], 
                       # "loss/danger_dist_max": o1[1], 
                       # "loss/safe_dist_min": o2[0],
                       # "loss/safe_dist_max": o2[1],})
    return


import scipy
from random import shuffle


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class GlobalReplayBuffer:
    """
    A buffer for storing trajectories experiences
    """

    def __init__(self, batch=64):
        self.obs_buf = []   
        self.batch = batch
        self.loader = None
        
    def store(self, obs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        expected keys: 'prev_free', 'next_free', 'prev_danger', 'next_danger', 'bvalue', 'action'
        """
        self.obs_buf.append(obs)
        
    def __getitem__(self, index):
        return self.obs_buf[index]
    
    def __len__(self):
        return len(self.obs_buf)

    def relabel(self):

        n_relabels = 0
        # back-prop the danger set
        for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf))[::-1]:
            if (idx < (len(self.obs_buf)-1)):
                self.obs_buf[idx]['returns'] = self.obs_buf[idx]['rewards'] + GAMMA * self.obs_buf[idx+1]['returns']

            else:
                self.obs_buf[idx]['returns'] = self.obs_buf[idx]['rewards']

        return n_relabels
    
    def next_data(self):
        
        if SHARE_SAMPLE_ACROSS_UPDATE:
            return Batch.from_data_list([self.obs_buf[idx] for idx in np.random.choice(len(self.obs_buf), min(self.batch, len(self.obs_buf)),
                                        replace=False)])
        else:
            if self.loader is None:
                self.loader = GeoDataLoader(self, shuffle=True, batch_size=self.batch)
                self.loader.sampler.replacement = False
                self.iter = iter(self.loader)        

            try:
                return next(self.iter)
            except StopIteration:
                self.iter = iter(self.loader)
                return next(self.iter)        
    
    
class MyDataset(Dataset):
    def __init__(self, data, next_data):
        self.data = data
        
    def __getitem__(self, index):
        xA = self.data[index]
        xB = self.data[(index+1) if ((index+1) <len(self.data)) else 0]
        return xA, xB
    
    def __len__(self):
        return len(self.data)            


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB    


class TrajectoryReplayBuffer:
    
    def __init__(self, batch=64):
        self.dataset = MyDataset([], [])
        self.batch = batch
        self.loader = None
        
    def append(self, buffer):
        
        new_buff = GlobalReplayBuffer()
        for idx, obs in enumerate(buffer.obs_buf):
            clone_obs = obs.clone()
            self.dataset.data.append(clone_obs)
                
    def next_data(self):
        if SHARE_SAMPLE_ACROSS_UPDATE:
            return collate([self.dataset[idx] for idx in np.random.choice(len(self.dataset), min(self.batch, len(self.dataset)),
                              replace=False)])
        else:        
            if self.loader is None:
                self.loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch, collate_fn=collate)
                self.loader.sampler.replacement = False
                self.iter = iter(self.loader)

            try:
                return next(self.iter)
            except StopIteration:
                self.iter = iter(self.loader)
                return next(self.iter)
            
            
@torch.no_grad()
def choose_action(actor, env, action_noise=0):
    action = actor(env._get_obs(**OBS_CONFIG).to(device)).data.cpu().numpy()
    action = action + np.random.randn(*action.shape) * action_noise
    action = np.clip(action, -1, 1)
    return action


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)


@torch.no_grad()
def infer(env, actor, max_episode_length=256,
          verbose=False, seed=0, stop_at_collision=False, 
          need_gif=None, return_safe_rate=False, **kwargs):
    
    if verbose:
        print('----------------------------------------')
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    paths = [deepcopy(env.world.agents)]
    total_trans=0; n_danger=0; collided=np.zeros(env.num_agents).astype(bool);unsafety_rate=[]

    while True:
        a = choose_action(actor=actor, env=env, action_noise=0)
        next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)
        
        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
        unsafety_rate.append(next_danger)
        if verbose:
            print(total_trans, np.where(next_danger))
            
        total_trans += 1

        paths.append(deepcopy(env.world.agents))

        if np.any(next_danger) and stop_at_collision:
            break        
        
        if done or (total_trans >= max_episode_length):
            break
            
    if (need_gif is not None):
        env.save_fig(paths, env.world.agent_goals, env.world.obstacles, need_gif[:-4]+'_'+str(np.any(collided))+'_'+str(done)+need_gif[-4:])

    if return_safe_rate:
        return collided, done, paths, np.mean(unsafety_rate)
    else:
        return collided, done, paths


if __name__ == '__main__':
    
    wandb.init(
        project=project_name+'_macbf',
        config={
            "env_config": ENV_CONFIG,
            "n_candidates": n_candidates,
            "batch": BATCH,
            "n_traj": N_TRAJ,
            "n_iter": N_ITER,
            "threshold": THRESHOLD,
            "n_evaluate": N_EVALUATE,
            "patience": PATIENCE,
            "lr": LR,
            "hidden_size": HIDDEN_SIZE,
            "steer": STEER,
            "relabel": RELABEL,
            "explore_way": EXPLORE_WAY,
            "nominal_way": NOMINAL_WAY,
            "decay_relabel": DECAY_RELABEL,
            "scheduler": USE_SCHEDULER,
            "optimizer": OPTIMIZER,
            "min_explore_eps": MIN_EXPLORE_EPS,
            "max_explore_eps": MAX_EXPLORE_EPS,
            "decay_explore_rate": DECAY_EXPLORE_RATE,
            "decay_nominal_rate": DECAY_NOMINAL_RATE,
            "potential_obs": POTENTIAL_OBS,
            "train_on_hard": TRAIN_ON_HARD,
            "refine": REFINE_EPS,
            "relabel_only_agent": RELABEL_ONLY_AGENT,
            "all_lie": ALL_LIE,
            "lie_derive_safe": LIE_DERIVE_SAFE,
            "only_boundary": ONLY_BOUNDARY,
            "polyak": POLYAK,
            "pe_dim": PE_DIM,
            "fix_env": FIX_ENV,
            "max_visit_time": MAX_VISIT_TIME,
            "clip_norm": CLIP_NORM,
            "dynamic_relabel": DYNAMIC_RELABEL,         
            "update_freq": UPDATE_FREQ,
            "min_lr": MIN_LR,
            "danger_threshold": DANGER_THRESHOLD,
            "all_explore": ALL_EXPLORE,
            "safe_explore": SAFE_EXPLORE,
            "danger_explore": DANGER_EXPLORE,
            "share_sample_across_update": SHARE_SAMPLE_ACROSS_UPDATE,
        },
        name=version_name.replace('_', '/'),)
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
        
    # generate valid data
    valid_dataset = []
    for _ in tqdm(range(N_VALID_DATASET)):
        env = create_env()
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))


    Env = Env
    env = create_env()
    critic = create_network(mode='critic')
    actor = create_network(mode='actor')
    
    data = env._get_obs(**OBS_CONFIG)
    data['action'] = torch.zeros(env.num_agents, env.action_dim)
    
    with torch.no_grad():
        data.to(device)
        print(critic(data))
        print(actor(data))
        
#     # Magic
#     wandb.watch(critic, log_freq=100, idx=0)                  
#     wandb.watch(actor, log_freq=100, idx=1)                  

    if OPTIMIZER=='SGD':
        qoptimizer = torch.optim.SGD(critic.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
        pioptimizer = torch.optim.SGD(actor.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    elif OPTIMIZER=='Adam':
        qoptimizer = torch.optim.Adam(critic.parameters(), lr=LR, weight_decay=1e-8)
        pioptimizer = torch.optim.Adam(actor.parameters(), lr=LR, weight_decay=1e-8)
    else:
        assert False
    if USE_SCHEDULER:
        qscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(qoptimizer, factor=0.5, patience=PATIENCE, min_lr=MIN_LR)
        pischeduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pioptimizer, factor=0.5, patience=PATIENCE, min_lr=MIN_LR)

    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*N_EVALUATE
    uncollide_rates = [1.]*N_EVALUATE
    success_rates = [0.]*N_EVALUATE
    nominal_eps = 1.0
    explore_eps = 1.0
    pointer = 0

    trajs = defaultdict(list)
    bbuf_traj = TrajectoryReplayBuffer(BATCH)
    bbuf_free = GlobalReplayBuffer(BATCH)
    bbuf_danger = GlobalReplayBuffer(BATCH)

    for epoch_i in range(N_TRAJ):

        t0 = time()
        if epoch_i < N_WARMUP:
            explore_eps = 0.
        elif EXPLORE_WAY=='cyclic':
            explore_eps = (MAX_EXPLORE_EPS-(MAX_EXPLORE_EPS-MIN_EXPLORE_EPS)*((epoch_i-N_WARMUP) % 100)/100.)  # if ((epoch_i % 200) < 100) else 0.
        elif EXPLORE_WAY=='linear':
            explore_eps = np.clip(MAX_EXPLORE_EPS - DECAY_EXPLORE_RATE * ((epoch_i-N_WARMUP) // N_VALID), MIN_EXPLORE_EPS, MAX_EXPLORE_EPS)
        elif EXPLORE_WAY=='exponential':
            explore_eps = np.clip(MAX_EXPLORE_EPS * (DECAY_EXPLORE_RATE ** (((epoch_i-N_WARMUP) // N_VALID))), MIN_EXPLORE_EPS, MAX_EXPLORE_EPS)
        else:
            assert False

        if epoch_i < N_WARMUP:
            nominal_eps = 1.
        elif NOMINAL_WAY=='linear':
            nominal_eps = np.clip(1. - DECAY_NOMINAL_RATE * ((epoch_i-N_WARMUP) // N_VALID), 0, 1.)
        elif NOMINAL_WAY=='exponential':
            nominal_eps = DECAY_NOMINAL_RATE ** (1e-5 + ((epoch_i-N_WARMUP) // N_VALID))
        else:
            assert False

        if nominal_eps < 0.01:
            nominal_eps = 0

        if DECAY_RELABEL:
            relabel_eps = 1 - explore_eps # (epoch_i // N_VALID) / (N_TRAJ // N_VALID)
        else:
            relabel_eps = REFINE_EPS

        torch.manual_seed(epoch_i)
        random.seed(epoch_i)
        np.random.seed(epoch_i)        
        
        env = create_env()

        total_trans=0; n_danger=0; threshold=THRESHOLD; collided=np.zeros(env.num_agents).astype(bool); rewards=[];
        thresholds = [threshold]*env.num_agents;
        bbuf = GlobalReplayBuffer(BATCH)

        while True:
            a = choose_action(actor=actor, env=env)
            
            a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim)) 
            dists = env.potential_field(a_all, K1 = 1e-1, K2 = -3e-2, ignore_agent=True)
            
            next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)

            info['oracle_action'] = torch.FloatTensor(a_all[np.arange(env.num_agents), np.argmin(dists, axis=-1), :])
            bbuf.store(info.clone())
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if np.any(next_danger):
                collided = collided | next_danger

            total_trans += 1
            pointer += 1
            n_danger += np.array(next_danger).sum()
            rewards.append(rw)

            if done or (total_trans >= max_episode_length):
                bbuf.obs_buf[-1]['finished'] = torch.tensor([True for _ in range(env.num_agents)])
                break
            else:
                bbuf.obs_buf[-1]['finished'] = torch.tensor([False for _ in range(env.num_agents)])

        for obs in bbuf.obs_buf:
            if obs['prev_danger'].bool().any():
                bbuf_danger.store(obs.clone())
            else:
                bbuf_free.store(obs.clone())
            
                
        unsafe_rates.append(collided.mean())
        unsafe_rates.pop(0)
        uncollide_rates.append(np.any(collided))
        uncollide_rates.pop(0)    
        success_rates.append(done and (not np.any(collided)))
        success_rates.pop(0)
        running_unsafe_rate = np.mean(unsafe_rates)
        
        wandb.log({'time/data_collection': time()-t0})           

        if (epoch_i > N_WARMUP) and (epoch_i % (N_VALID) == (N_VALID-1)):

            torch.save(actor.state_dict(), ACTOR_PATH.replace('.pt', '_{0:d}.pt'.format(epoch_i // N_VALID)))

            valid_loss = 0
            valid_success = 0
            valid_length = 0
            for v_idx, data in enumerate(valid_dataset):
                torch.manual_seed(2**32-1-v_idx)
                random.seed(2**32-1-v_idx)
                np.random.seed(2**32-1-v_idx)
                env = create_env()
                # env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done, gifs, _ = infer(env, actor, need_gif=None, return_safe_rate=True)
                valid_loss += np.mean(collided)
                valid_success += (done and (not np.any(collided)))
                valid_length += len(gifs)
            
            if USE_SCHEDULER:
                qscheduler.step(valid_loss/len(valid_dataset)+100*(1-valid_success/len(valid_dataset)))
                pischeduler.step(valid_loss/len(valid_dataset)+100*(1-valid_success/len(valid_dataset)))
            
            wandb.log({"valid/valid loss": valid_loss/len(valid_dataset),
                       "valid/valid length": valid_length/len(valid_dataset),
                       "valid/valid success": valid_success/len(valid_dataset),})                

        if epoch_i == 9:         
            print(actor)        
            print(critic)           
        
        if running_unsafe_rate < best_unsafe_rate:
            best_unsafe_rate = running_unsafe_rate
            torch.save(actor.state_dict(), ACTOR_PATH)  
            
        t0 = time()
        while len(bbuf_free.obs_buf) > N_BUFFER:
            bbuf_free.obs_buf.pop(0)
        while len(bbuf_danger.obs_buf) > N_BUFFER:
            bbuf_danger.obs_buf.pop(0)            
        critic.train()
        actor.train()
        pioptimizer.zero_grad()
        qoptimizer.zero_grad()
        if ((epoch_i // 10) % 2) == 0:
            train_barrier(critic, actor, qoptimizer, bbuf_free, bbuf_danger, env, 20)
        else:
            train_barrier(critic, actor, pioptimizer, bbuf_free, bbuf_danger, env, 20)
        wandb.log({'time/training': time()-t0})

        torch.save(actor.state_dict(), ACTOR_PATH.replace('.pt', '_current.pt'))

            
        if epoch_i < N_WARMUP:
            torch.save(actor.state_dict(), ACTOR_PATH.replace('.pt', '_warmup.pt'))
                

        wandb.log({"loss/lr": qoptimizer.param_groups[0]['lr'],
                   "uncollide_rates": np.mean(uncollide_rates),
                   "success_rates": np.mean(success_rates),
                   "running_unsafe_rate": running_unsafe_rate,
                   "explore_eps": explore_eps,
                   "nominal_eps": nominal_eps,
                   "relabel_prob": relabel_eps,
                   "n_trans": total_trans,
                   "size/epoch_i": epoch_i,
                   "size/buffer_free": len(bbuf_free.obs_buf),
                   "size/buffer_danger": len(bbuf_danger.obs_buf),
                   "rewards": np.mean(np.sum(rewards, axis=0))})
        
