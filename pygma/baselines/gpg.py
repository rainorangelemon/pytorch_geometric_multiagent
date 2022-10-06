import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER
from environment.gym_drone import DroneEnv
import torch.nn.functional  as F

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
from baselines.gpg_config import *

POLYAK = 0.995

Env = eval(env_name)
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_'+version_name+'.txt'
ACTOR_PATH = name_dict['db'].replace('.pt', '_'+version_name+'_actor.pt')
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


def create_network(mode):
    env = create_env()
    if ENV_CONFIG['PROB'][1] == 0:
        nn = RLNet(HIDDEN_SIZE, None if mode=='critic' else 2*env.action_dim, use_tanh=False, keys=['agent', 'goal'], pos_encode=PE_DIM)
    else:
        nn = RLNet(HIDDEN_SIZE, None if mode=='critic' else 2*env.action_dim, use_tanh=False, keys=['agent', 'obstacle', 'goal'], pos_encode=PE_DIM)
    nn.to(device)
    return nn


def train_barrier(critic, target_critic, actor, target_actor, qoptimizer, piotimizer, buf_traj, n_iter):
    
    def compute_loss_q(data, next_data):
        q = critic(data)

        with torch.no_grad():
            next_data = next_data.clone()
            next_data['action'] = target_actor(next_data)
            q_pi_targ = target_critic(next_data)
            backup = data['rewards'] + GAMMA * (1 - data['finished'].float()) * q_pi_targ

        loss_q = ((q - backup)**2).mean()
        wandb.log({"loss/q_pi_targ": q_pi_targ,
                   "loss/q": q,
                   "loss/finished": data['finished'].float().mean()})
        return loss_q

    def compute_loss_pi(data):
        data = data.clone()
        data['action'] = actor(data)
        q_pi = critic(data)
        return -q_pi.mean()    

    for _ in range(n_iter):
        data, next_data = buf_traj.next_data()
        data = data.to(device)
        next_data = next_data.to(device)    
        qoptimizer.zero_grad()
        loss_q = compute_loss_q(data, next_data)
        loss_q.backward()
        qoptimizer.step()    

        for p in critic.parameters():
            p.requires_grad = False

        pioptimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pioptimizer.step()

        for p in critic.parameters():
            p.requires_grad = True

        wandb.log({"loss/loss_pi": loss_pi,
                   "loss/loss_q": loss_q,})

        with torch.no_grad():
            for p, p_targ in zip(critic.parameters(), target_critic.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)

            for p, p_targ in zip(actor.parameters(), target_actor.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)
    
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
def choose_action_greedy(actor, env):
    action_dim = env.action_dim
    action_sigma = actor(env._get_obs(**OBS_CONFIG).to(device))
    action = action_sigma[:, :action_dim].tanh().data.cpu().numpy()
    return action


def choose_action_gradient(actor, env):
    action_dim = env.action_dim
    action_sigma = actor(env._get_obs(**OBS_CONFIG).to(device))
    mu = action_sigma[:, :action_dim].tanh()
    std = torch.exp(action_sigma[:, action_dim:].clip(-20, 2))
    pi_distribution = Normal(mu, std)
    pi_action = pi_distribution.rsample()

    logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
    
    return pi_action.clip(-1, 1).data.cpu().numpy(), logp_pi.sum()


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)


@torch.no_grad()
def infer(env, actor, max_episode_length=256,
          verbose=False, seed=0, stop_at_collision=False, need_gif=None):
    
    if verbose:
        print('----------------------------------------')
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    paths = [deepcopy(env.world.agents)]
    total_trans=0; n_danger=0; collided=np.zeros(env.num_agents).astype(bool);

    while True:
        a = choose_action_greedy(actor=actor, env=env)
        next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)
        
        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
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

    return collided, done, paths


if __name__ == '__main__':
    
    wandb.init(
        project=project_name,
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
    
    target_critic = deepcopy(critic)
    target_actor = deepcopy(actor)
    
    for p in target_critic.parameters():
        p.requires_grad = False
    for p in target_actor.parameters():
        p.requires_grad = False            

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
        
        bbuf = GlobalReplayBuffer()
        env = create_env()

        total_trans=0; n_danger=0; threshold=THRESHOLD; collided=np.zeros(env.num_agents).astype(bool); rewards=[];
        thresholds = [threshold]*env.num_agents; action_probs=[];

        while True:
            a, action_prob = choose_action_gradient(actor=actor, env=env)
            
            next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)

            bbuf.store(info.clone())
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if np.any(next_danger):
                collided = collided | next_danger

            total_trans += 1
            pointer += 1
            n_danger += np.array(next_danger).sum()
            action_probs.append(action_prob)
            rewards.append(rw)

            if (np.any(collided)) or done or (total_trans >= max_episode_length):
                bbuf.obs_buf[-1]['finished'] = torch.tensor([True for _ in range(env.num_agents)])
                break
            else:
                bbuf.obs_buf[-1]['finished'] = torch.tensor([False for _ in range(env.num_agents)])

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
                env = create_env(num_agents=len(data[2]))
                env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done, gifs = infer(env, actor, need_gif=None)
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
        
        if (epoch_i % N_EVALUATE) == (N_EVALUATE-1) and (running_unsafe_rate!=0):
            if running_unsafe_rate < best_unsafe_rate:
                best_unsafe_rate = running_unsafe_rate
                torch.save(actor.state_dict(), ACTOR_PATH)  
            
        t0 = time()
        critic.train()
        actor.train()
        bbuf.relabel()
        returns = torch.FloatTensor([b['returns'].sum() for b in bbuf.obs_buf])
        returns = torch.nn.functional.normalize(returns-returns.mean(), dim=0)
        action_probs = torch.stack(action_probs)
        pioptimizer.zero_grad()
        loss = (action_probs*returns.to(device)).mean()
        (-loss).backward()
        wandb.log({'loss/loss': loss,
                   'loss/log_prob': action_probs.mean(),
                   'loss/returns': returns.mean(),})
        pioptimizer.step()
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
                   "rewards": np.mean(np.sum(rewards, axis=0))})
        
