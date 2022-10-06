from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER
from environment.gym_drone import DroneEnv
from environment.gym_dynamic_dubins_multi import MultiDynamicDubinsEnv, STEER

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

if __name__ == '__main__':
    from configs.multi_dynamic_dubins.v3 import *
else:
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
    if FIX_ENV:
        env = Env(**env_config)
        env.world.obstacles = np.array([[1.5, 1.5]])
        env.world.agents = np.array([[0.5, 0.5, 0]+[0]*6,
                                     [2.5, 2.5, 0]+[0]*6,
                                     [0.5, 2.5, 0]+[0]*6])
        env.world.agent_goals = np.array([[2.5, 2.5, 1.5]+[0]*6,
                                          [0.5, 0.5, 0.5]+[0]*6,
                                          [2.5, 0.5, 1]+[0]*6])
    else:
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
            # if (np.linalg.norm(env.world.agents[:,:env.space_dim] - env.world.agent_goals[:,:env.space_dim], axis=-1).min() >= min_dist):
            #     break
    return env


def train_barrier(bnn, swa_bnn, optimizer, buf_traj, buf_free, buf_danger, buf_agent, buf_obstacle, n_iter=10):
    
    # Set up function for computing value loss
    def compute_bloss(bnn, data):
        value = bnn(data)
        
        bloss1 = ((-value).relu())*data['next_free'] / (1e-9 + (data['next_free']).sum())
        bloss2 = ((1e-1+value).relu())*data['next_danger'] / (1e-9 + (data['next_danger'].sum()))
        bloss = bloss1.sum() + bloss2.sum()
        return bloss

    def compute_dloss(bnn, data, next_data):
        value = bnn(data)
        next_value = bnn(next_data)
        
        if not ALL_LIE:
            deriv = next_value-value+0.1*value
            dloss = ((-deriv+1e-2).relu())*data['next_free']*next_data['next_free']
            dloss = dloss.sum() / (1e-9 + (data['next_free']*next_data['next_free']).sum())
        else:
            deriv = next_value-value+0.1*value
            dloss = ((-deriv+1e-2).relu())
            dloss = dloss.mean()

        bloss1 = ((-value).relu())*data['next_free'] / (1e-9 + (data['next_free']).sum())
        bloss2 = ((-next_value).relu())*next_data['next_free'] / (1e-9 + (next_data['next_free'].sum()))
        bloss = bloss1.sum() + bloss2.sum()         
            
        return dloss + bloss

    for i in range(n_iter):
        if (buf_traj is not None) and len(buf_traj.dataset):
            data, next_data = buf_traj.next_data()
            optimizer.zero_grad()
            data = data.to(device)
            next_data = next_data.to(device)
            dloss = compute_dloss(bnn, data, next_data)
        else:
            dloss = 0

        if (buf_free is not None) and len(buf_free):
            data = buf_free.next_data()
            optimizer.zero_grad()
            data = data.to(device)
            bloss_f = compute_bloss(bnn, data)
        else:
            bloss_f = 0
            
        if (buf_danger is not None) and len(buf_danger):
            data = buf_danger.next_data()
            optimizer.zero_grad()
            data = data.to(device)
            bloss_d = compute_bloss(bnn, data)
        else:
            bloss_d = 0        
        
        if (buf_agent is not None) and len(buf_agent):
            data = buf_agent.next_data()
            optimizer.zero_grad()
            data = data.to(device)
            closs_a = compute_bloss(bnn, data)   
        else:
            closs_a = 0
        
        if (buf_obstacle is not None) and len(buf_obstacle):
            data = buf_obstacle.next_data()
            optimizer.zero_grad()
            data = data.to(device)
            closs_o = compute_bloss(bnn, data)
        else:
            closs_o = 0
        
        (bloss_f + bloss_d + dloss + closs_a + closs_o).backward()        
        if CLIP_NORM:
            torch.nn.utils.clip_grad_norm_(bnn.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"loss/bloss_f": bloss_f,
                   "loss/bloss_d": bloss_d,
                   "loss/dloss": dloss,
                   "loss/closs_a": closs_a,
                   "loss/closs_o": closs_o})
        
        if swa_bnn is not None:
            with torch.no_grad():
                for p, swa_p in zip(bnn.parameters(), swa_bnn.parameters()):
                    swa_p.data.mul_(POLYAK)
                    swa_p.data.add_((1 - POLYAK) * p.data)    
    return


# create replay buffer
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

    def relabel(self, prob):

        n_relabels = 0
        # back-prop the danger set
        for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf))[::-1]:
            if (idx < (len(self.obs_buf)-1)):
                self.obs_buf[idx]['next_danger'] = self.obs_buf[idx+1]['prev_danger']
                self.obs_buf[idx]['next_free'] = self.obs_buf[idx+1]['prev_free']

            if RELABEL_ONLY_AGENT:
                obs['feasible'] = ((obs['feasible'] + obs['meet_obstacle']) >= 1).float()

            inadmissible = ((torch.rand(*(obs['next_danger'].shape)) < prob) * obs['next_danger'] * (1-obs['feasible']))

            obs['prev_danger'] = ((obs['prev_danger'] + inadmissible) >= 1).float()
            obs['prev_free'] = obs['prev_free'] * (1-obs['prev_danger'])
            n_relabels += inadmissible.sum()

        if ONLY_BOUNDARY:
            for idx, obs in list(zip(range(len(self.obs_buf)), self.obs_buf)):
                need = (1-obs['prev_danger'])*obs['next_danger'] + obs['meet_obstacle'] + obs['meet_agent']
                need = (need >= 1)
                obs['next_danger'][~need] = 0

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
        self.next_data = next_data
        
    def __getitem__(self, index):
        xA = self.data[index]
        xB = self.next_data[index]
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
        
    def append(self, buffer, collided):
        if np.all(collided) and (not ALL_LIE):
            return
        
        new_buff = GlobalReplayBuffer()
        for idx, obs in enumerate(buffer.obs_buf):
            clone_obs = obs.clone()
            clone_obs['next_free'] = (1-torch.FloatTensor(collided))*clone_obs['next_free']
            new_buff.obs_buf.append(clone_obs)
            if idx > 0:
                self.dataset.data.append(new_buff.obs_buf[-2])
                self.dataset.next_data.append(new_buff.obs_buf[-1]) 
                
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


class GatherReplayBuffer(Dataset):

    def __init__(self, bnn, dynamic_relabel=False, batch=64):
        self.dataset = []
        self.bnn = bnn
        self.dynamic_relabel = dynamic_relabel
        self.batch = batch
        self.loader = None
        
    def append(self, buffer):
        for o in buffer.obs_buf:
            self.dataset.append(o)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def next_data(self):
        if SHARE_SAMPLE_ACROSS_UPDATE:
            return Batch.from_data_list([self.dataset[idx] for idx in np.random.choice(len(self.dataset), min(self.batch, len(self.dataset)), 
                       replace=False)])   
        else:
            if self.loader is None:
                self.loader = GeoDataLoader(self.dataset, shuffle=True, batch_size=self.batch)
                self.loader.sampler.replacement = False
                self.iter = iter(self.loader)        

            try:
                return next(self.iter)
            except StopIteration:
                if self.dynamic_relabel:
                    for index in np.arange(len(self.dataset))[::-1]:
                        self.relabel(index, self.bnn)

                self.iter = iter(self.loader)
                return next(self.iter)
     
    @torch.no_grad()
    def relabel(self, idx, bnn):
        if self.dataset[idx]['finished']:
            return self.dataset[idx]
        else:
            data = self.dataset[idx]
            next_data = self.dataset[idx+1]
            next_danger = next_data['next_danger']
            with torch.no_grad():
                tensor_a = torch.zeros(size=(len(next_data['agent'].x), n_candidates, next_data['action'].shape[-1]), device=device).uniform_(-1, 1)
                vec = bnn.get_vec(next_data.clone().to(device))
                vec = vec.unsqueeze(1).repeat((1, n_candidates, 1))
                next_bvalue = bnn.get_field(vec, tensor_a)   
            suspicous = (next_bvalue<THRESHOLD).all(dim=-1).cpu().float()
            data['next_danger'] = ((suspicous*next_danger + data['meet_agent'] + data['meet_obstacle'])>=1).float()
            data['next_free'] = 1-data['next_danger']


@torch.no_grad() 
def eval_action(bnn, o, a):
    # size of a: (num_agents, n_action, action_dim)
    n_action = a.shape[1]

    input_ = o.clone().to(device)
    tensor_a = torch.FloatTensor(a).to(device)

    input_['action'] = tensor_a
    vec = bnn.get_vec(input_)
    vec = vec.unsqueeze(1).repeat((1, n_action, 1))
    bvalue = bnn.get_field(vec, tensor_a)
    return tensor_a.data.cpu().numpy(), bvalue.data.cpu().numpy()


@torch.no_grad() 
def eval_action_adaptive(bnn, o, a, thresholds):
    
    # size of a: (num_agents, n_action, action_dim)
    num_agents, n_action, action_dim = a.shape
    thresholds = torch.FloatTensor(thresholds)[o['agent'].n_id]
    
    batch = int(1000000000/(4*num_agents*(HIDDEN_SIZE+action_dim)))

    input_ = o.clone().to(device)
    tensor_a = torch.FloatTensor(a)

    input_['action'] = tensor_a
    vec = bnn.get_vec(input_)
    
    bvalue_final = -1 * torch.ones_like(tensor_a[...,0]).cpu()
    
    start_idx = 0
    while start_idx < n_action:
        need_agents = (bvalue_final.max(dim=-1)[0]) < thresholds
        if (~need_agents).all():
            break
        batch_adapt = int(batch*num_agents/need_agents.sum())
        sub_tensor_a = tensor_a[need_agents, start_idx:(start_idx+batch_adapt), :]
        sub_vec = vec[need_agents, :].unsqueeze(1).repeat((1, sub_tensor_a.shape[1], 1))
        bvalue = bnn.get_field(sub_vec, sub_tensor_a.to(device))
        bvalue_final[need_agents, start_idx:(start_idx+batch_adapt)] = bvalue.detach().data.cpu()
        del bvalue
        
        start_idx = start_idx + batch_adapt
            
    
    return tensor_a.data.cpu().numpy(), bvalue_final.data.cpu().numpy()


def choose_action(bnn, env, explore_eps, nominal_eps, spatial_prop, thresholds, n_action=None, decompose=None):
    if n_action is None:
        n_action = n_candidates

    if nominal_eps > 0:
        K1 = 1e-1
        K2 = -3e-2
    else:
        K1 = 0.
        K2 = -3e-2
    
    if decompose is None:
        o = env._get_obs(**OBS_CONFIG)
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_action, env.action_dim))
        a_refines, bvalues = eval_action(bnn, o, a_all)
    else:
        a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_action, env.action_dim))
        bvalues = np.ones(shape=(env.num_agents, n_action))
        if decompose == 'random':
            a_refines = a_all
            graphs = list(env._get_obs_random_k(**OBS_CONFIG_DECOMPOSE))
            dataset = Batch.from_data_list(graphs)
            
            a_all_all = torch.as_tensor(a_all)[dataset['agent'].n_id,:,:].float()
            _, decomposed_values = eval_action_adaptive(bnn, dataset, a_all_all, thresholds)
            bvalues = decomposed_values.reshape((len(graphs), env.num_agents, n_action)).min(axis=0)
            
            del dataset
            del decomposed_values
            del a_all_all
        
        elif decompose == 'group':
            r_graph = env._get_obs(rgraph_a=True, rgraph_o=False)
            prefer_center = set(np.random.permutation(env.num_agents))
            r_edge_index = r_graph['a_near_a'].edge_index
            edge_mask = torch.zeros_like(r_graph['a_near_a'].edge_index[0,:]).bool()
            while len(prefer_center):
                result = env._get_obs_group_k(np.random.permutation(list(prefer_center)), loop=False, clip=True)
                edge_mask = edge_mask | result[3]
                prefer_center = (prefer_center - result[2]) & set(r_edge_index[:,~edge_mask].unique().data.numpy())
                o = result[0]
                a_refines, bvalues_current = eval_action(bnn, o, a_all)  
                bvalues[np.array(list(result[1])),:] = np.minimum(bvalues[np.array(list(result[1])),:], 
                                                                  bvalues_current[np.array(list(result[1])),:])

        else:
            assert False       

    dists = env.potential_field(a_refines, K1=K1, K2=K2, ignore_agent=(nominal_eps <= 0))
    v = np.zeros(env.num_agents)
    a = np.zeros((env.num_agents, env.action_dim))
    feasibles = np.zeros(env.num_agents)
    evil_agents = set()
    for agent_id, a_refine, bvalue, dist, threshold in zip(np.arange(env.num_agents), a_refines, bvalues, dists, thresholds):

        feasibles[agent_id] = float(np.any(bvalue>DANGER_THRESHOLD))
        
        if np.random.rand() < nominal_eps:
            a[agent_id] = a_refine[np.argsort(dist)[0]]
            # feasibles[agent_id] = 1
            continue
            
        feasible_current = False
        if np.any(bvalue>threshold):
            feasible_current = True
        else:
            feasible_current = False            
            
        if (SAFE_EXPLORE and feasible_current) and (np.random.rand()<explore_eps):
            a_idx = np.random.choice(np.where(bvalue>threshold)[0])
            a[agent_id] = a_refine[a_idx]
            # feasibles[agent_id] = 1  # mask the random action            
            continue            
            
        if feasible_current:
            feasibles[agent_id] = 1
            prefered_idxs = np.argsort(dist)
            prefered_idxs = prefered_idxs[bvalue[prefered_idxs]>threshold]
            a_idx = prefered_idxs[0]
            a[agent_id] = a_refine[a_idx]
            v[agent_id] = bvalue[a_idx]
            continue

        if spatial_prop:
            local_evils = set()
            # find evil_agent
            local_o = o.clone()
            while True:
                local_o.to('cpu')
                edges = local_o['a_near_a'].edge_index
                neighbor_edges = edges[1]==agent_id
                if neighbor_edges.sum()==0:
                    break

                first_edge = torch.where(neighbor_edges)[0][0]
                mask = (torch.arange(edges.shape[1])==first_edge)
                local_evils.add(int(edges[0, first_edge]))
                local_o['a_near_a'].edge_index = edges[:,~mask]
                local_o['a_near_a'].edge_attr = local_o['a_near_a'].edge_attr[~mask,:]

                local_a_refines, local_bvalues = eval_action(bnn, local_o, a_all)

                if np.any(local_bvalues[agent_id]>threshold):
                    evil_agents = evil_agents | local_evils
                    break
                    
        a[agent_id] = a_refine[np.argmax(bvalue)]
        v[agent_id] = bvalue[np.argmax(bvalue)]
        if DANGER_EXPLORE and (not feasible_current):
            a[agent_id] = a[agent_id] + (explore_eps!=0)*0.1*np.random.randn(*(a[agent_id].shape))
            a[agent_id] = a[agent_id].clip(-1, 1)

    for evil_agent in evil_agents:
        a_refine, bvalue = a_refines[evil_agent], bvalues[evil_agent]
        a[evil_agent] = a_refine[np.argmax(bvalue)]
        v[evil_agent] = bvalue[np.argmax(bvalue)]
        
    if ALL_EXPLORE and (np.random.rand()<explore_eps):
        a = a + (explore_eps!=0)*0.1*np.random.randn(*(a.shape))
        a = a.clip(-1, 1)
        
    return a, v, feasibles, evil_agents


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)


@torch.no_grad()
def infer(env, bnn, threshold=None, max_episode_length=256, 
          n_action=None,
          verbose=False, seed=0, stop_at_collision=False, 
          stop_at_done=True,
          spatial_prop=None, need_gif=None, 
          decompose=None, lie_derive_safe=None,
          only_show_collide=False):
    
    if spatial_prop is None:
        spatial_prop = SPATIAL_PROP
        
    if n_action is None:
        n_action = n_candidates
        
    if lie_derive_safe is None:
        lie_derive_safe = LIE_DERIVE_SAFE
    
    if threshold is None:
        threshold=THRESHOLD
    if verbose:
        print('----------------------------------------')
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    paths = [deepcopy(env.world.agents)]
    total_trans=0; n_danger=0; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool); thresholds=np.array([threshold]*env.num_agents)

    while True:
        a, v, feasibles, evil_agents = choose_action(bnn=bnn, env=env, explore_eps=0, 
                                                     nominal_eps=0, 
                                                     spatial_prop=spatial_prop, 
                                                     thresholds=thresholds,
                                                     n_action=n_action,
                                                     decompose=decompose)
        next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)
        
        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
        if verbose:
            print(total_trans, v.min(axis=-1), v.max(axis=-1), np.where(v<=thresholds), np.where(next_danger), evil_agents)
            
        total_trans += 1
        if lie_derive_safe:
            thresholds = 0.9*v+1e-2

        paths.append(deepcopy(env.world.agents))

        if np.any(next_danger) and stop_at_collision:
            break        
        
        if (stop_at_done and done) or (total_trans >= max_episode_length):
            break
            
    if (need_gif is not None):
        if (not only_show_collide) or (np.any(collided)):
            env.save_fig(paths, env.world.agent_goals, env.world.obstacles, need_gif[:-4]+'_'+str(np.any(collided))+'_'+str(done)+need_gif[-4:])

    return collided, done, paths


if __name__ == '__main__':
    
    run = wandb.init(
        project=project_name,
        config={
            "env_config": ENV_CONFIG,
            "n_candidates": n_candidates,
            "batch": BATCH,
            "n_traj": N_TRAJ,
            "n_iter": N_ITER,
            "threshold": THRESHOLD,
            "n_traj_per_update": N_TRAJ_PER_UPDATE,
            "n_dynamic_buffer_free": N_DYNAMIC_BUFFER_FREE,
            "n_dynamic_buffer_danger": N_DYNAMIC_BUFFER_DANGER,
            "n_traj_buffer": N_TRAJ_BUFFER,
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
            "cbuf": N_CBUF,
            "update_freq": UPDATE_FREQ,
            "min_lr": MIN_LR,
            "danger_threshold": DANGER_THRESHOLD,
            "all_explore": ALL_EXPLORE,
            "safe_explore": SAFE_EXPLORE,
            "danger_explore": DANGER_EXPLORE,
        },
        name=version_name,)
    
    run.log_code(".", include_fn=lambda path: ("drone_v34.py" in path) or ("gym_drone.py" in path))
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # generate training data
    dataset = []
    for _ in tqdm(range(N_DATASET)):
        env = create_env()
        dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])
        
    # generate valid data
    valid_dataset = []
    for _ in tqdm(range(N_VALID_DATASET)):
        torch.manual_seed(2**32-1-_)
        random.seed(2**32-1-_)
        np.random.seed(2**32-1-_)
        env = create_env()
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))


    Env = Env
    env = create_env()
    bnn = create_network()
    swa_bnn = None

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    if OPTIMIZER=='SGD':
        boptimizer = torch.optim.SGD(bnn.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    elif OPTIMIZER=='Adam':
        boptimizer = torch.optim.Adam(bnn.parameters(), lr=LR, weight_decay=1e-8)
    else:
        assert False
    if USE_SCHEDULER:
        bscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(boptimizer, factor=0.5, patience=PATIENCE, min_lr=MIN_LR)

    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*N_EVALUATE
    uncollide_rates = [1.]*N_EVALUATE
    success_rates = [0.]*N_EVALUATE
    nominal_eps = 1.0
    explore_eps = 1.0

    trajs = defaultdict(list)
    # open(TXT_NAME, 'w+').close()
    cbuf_obstacle = GlobalReplayBuffer(BATCH)
    cbuf_agent = GlobalReplayBuffer(BATCH)
    cbuf_dynamic_danger = GlobalReplayBuffer(BATCH)
    cbuf_dynamic_free = GlobalReplayBuffer(BATCH)
    bbuf_traj = TrajectoryReplayBuffer(BATCH)
    bbuf_gather = GatherReplayBuffer(bnn=swa_bnn, dynamic_relabel=DYNAMIC_RELABEL, batch=BATCH)
    n_new_data = 0

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

        if (epoch_i % len(dataset) == 0):
            shuffle(dataset)

        torch.manual_seed(epoch_i)
        random.seed(epoch_i)
        np.random.seed(epoch_i)        
        
        bbuf = GlobalReplayBuffer()      
        env = create_env()
        dataset[epoch_i%len(dataset)][-1] += 1; visit_time = 0
        if TRAIN_ON_HARD:
            if epoch_i > N_WARMUP:
                env.world.obstacles, env.world.agent_goals, env.world.agents, visit_time = deepcopy(dataset[epoch_i%len(dataset)])

        total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool); 
        volumes=[]; n_relabels=0; n_evils = []
        thresholds = [threshold]*env.num_agents

        while True:
            a, v, feasibles, evil_agents = choose_action(
                                            bnn=bnn,
                                            env=env,
                                            explore_eps=explore_eps, 
                                            nominal_eps=max(nominal_eps, int(epoch_i<N_WARMUP)), 
                                            spatial_prop=SPATIAL_PROP,
                                            n_action=n_candidates,
                                            thresholds=thresholds)
            n_evils.append(len(evil_agents))
            no_feasible += (env.num_agents - np.sum(feasibles))
            next_o, rw, done, info = env.step(a, obs_config=OBS_CONFIG)

            info['feasible'] = torch.FloatTensor(feasibles)
            bbuf.store(info.clone())
            prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if np.any(next_danger):
                collided = collided | next_danger

            total_trans += 1
            n_danger += np.array(next_danger).sum()
            volumes.append((v>threshold).mean())
            if LIE_DERIVE_SAFE:
                thresholds = 0.9*v+1e-2

            if done or (total_trans >= max_episode_length):
                bbuf.obs_buf[-1]['finished'] = True
                
                if (n_danger==0) or (visit_time > MAX_VISIT_TIME):
                    # only preserve the hard envs
                    dataset.pop(epoch_i%len(dataset))
                    env = create_env()
                    dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])

                if (n_danger!=0):
                    if RELABEL:
                        if epoch_i > N_WARMUP:
                            n_relabels = bbuf.relabel(relabel_eps) 

                for info in bbuf.obs_buf:
                    meet_agent = info['meet_agent'].data.cpu().numpy().astype(bool)                        
                    prev_agent = info['prev_agent'].data.cpu().numpy().astype(bool)                        
                    meet_obstacle = info['meet_obstacle'].data.cpu().numpy().astype(bool)                        
                    prev_obstacle = info['prev_obstacle'].data.cpu().numpy().astype(bool)                                            
                    prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)                        
                    next_danger = info['next_danger'].data.cpu().numpy().astype(bool)                        
                    next_free = info['next_free'].data.cpu().numpy().astype(bool)                         
                    prev_free = info['prev_free'].data.cpu().numpy().astype(bool)
                    feasible = info['feasible'].data.cpu().numpy().astype(bool)
                    if (meet_agent).any():
                        data = info.clone()
                        data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_danger'] = torch.FloatTensor(meet_agent)
                        cbuf_agent.store(data)  

                    if (meet_obstacle).any():
                        data = info.clone()
                        data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_danger'] = torch.FloatTensor(meet_obstacle)
                        cbuf_obstacle.store(data)
                        
                    if (next_danger).any():
                        data = info.clone()
                        data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_danger'] = torch.FloatTensor(next_danger)
                        cbuf_dynamic_danger.store(data)   
                        n_new_data += 1

                    if ((~feasible) & next_free).any():
                        data = info.clone()
                        data['prev_free'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_danger'] = torch.FloatTensor([False]*env.num_agents)
                        data['next_free'] = torch.FloatTensor((~feasible) & next_free)
                        cbuf_dynamic_free.store(data)
                        n_new_data += 1
                
                if not (collided.all()):
                    bbuf_traj.append(bbuf, collided)
                bbuf_gather.append(bbuf)
                break
                
            else:
                bbuf.obs_buf[-1]['finished'] = False
                
        unsafe_rates.append(collided.mean())
        unsafe_rates.pop(0)
        uncollide_rates.append(np.any(collided))
        uncollide_rates.pop(0)    
        success_rates.append(done and (not np.any(collided)))
        success_rates.pop(0)
        running_unsafe_rate = np.mean(unsafe_rates)
        
        wandb.log({'time/data_collection': time()-t0})

        while len(bbuf_gather.dataset) > N_BUFFER:
            bbuf_gather.dataset.pop(0)
            
        while len(bbuf_traj.dataset) > N_TRAJ_BUFFER:
            bbuf_traj.dataset.data.pop(0)
            bbuf_traj.dataset.next_data.pop(0)
        
        while len(cbuf_agent.obs_buf) > N_CBUF:
            cbuf_agent.obs_buf.pop(0)
            
        while len(cbuf_obstacle.obs_buf) > N_CBUF:
            cbuf_obstacle.obs_buf.pop(0)

        if (epoch_i > N_WARMUP) and (epoch_i % (N_VALID) == (N_VALID-1)):

            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_{0:d}.pt'.format(epoch_i // N_VALID)))

            valid_loss = 0
            valid_success = 0
            valid_length = 0
            for v_idx, data in enumerate(valid_dataset):
                env = create_env(num_agents=len(data[2]))
                env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done, gifs = infer(env, bnn, need_gif=None, seed=(epoch_i+v_idx))
                valid_loss += np.mean(collided)
                valid_success += (done and (not np.any(collided)))
                valid_length += len(gifs)
            
            if USE_SCHEDULER:
                bscheduler.step(running_unsafe_rate)  # valid_loss/len(valid_dataset)+100*(1-valid_success/len(valid_dataset)))
            
            wandb.log({"valid/valid loss": valid_loss/len(valid_dataset),
                       "valid/valid length": valid_length/len(valid_dataset),
                       "valid/valid success": valid_success/len(valid_dataset),})                

        if epoch_i == 9:         
            print(bnn)        
        
        if running_unsafe_rate < best_unsafe_rate:
            best_unsafe_rate = running_unsafe_rate
            torch.save(bnn.state_dict(), BMODEL_PATH)        
        
        if (epoch_i > N_WARMUP) and (epoch_i % (N_TRAJ_PER_UPDATE) == (N_TRAJ_PER_UPDATE-1)):
            t0 = time()
            bnn.train()
            if (swa_bnn is None) and (POLYAK != 0):
                swa_bnn = create_network()
                with torch.no_grad():
                    for p, swa_p in zip(bnn.parameters(), swa_bnn.parameters()):
                        swa_p.data = deepcopy(p.data)
                bbuf_gather.bnn = swa_bnn

#             train_dynamic_buffer = (n_new_data >= BATCH)
            
#             if train_dynamic_buffer:
            if len(cbuf_dynamic_free.obs_buf) > (N_DYNAMIC_BUFFER_FREE+BATCH):
                new_buf = [cbuf_dynamic_free.obs_buf[idx+N_DYNAMIC_BUFFER_FREE] for idx in np.random.choice(len(cbuf_dynamic_free.obs_buf)-N_DYNAMIC_BUFFER_FREE, BATCH, replace=False)]
                cbuf_dynamic_free.obs_buf = cbuf_dynamic_free.obs_buf[:N_DYNAMIC_BUFFER_FREE] + new_buf
                
            if len(cbuf_dynamic_danger.obs_buf) > (N_DYNAMIC_BUFFER_DANGER+BATCH):
                new_buf = [cbuf_dynamic_danger.obs_buf[idx+N_DYNAMIC_BUFFER_DANGER] for idx in np.random.choice(len(cbuf_dynamic_danger.obs_buf)-N_DYNAMIC_BUFFER_DANGER, BATCH, replace=False)]
                cbuf_dynamic_danger.obs_buf = cbuf_dynamic_danger.obs_buf[:N_DYNAMIC_BUFFER_DANGER] + new_buf                

            train_barrier(bnn, swa_bnn, boptimizer, 
                      bbuf_traj, 
                      cbuf_dynamic_free,
                      cbuf_dynamic_danger,
                      cbuf_agent, 
                      cbuf_obstacle, 
                      n_iter=N_ITER)
            
            # shuffle(cbuf_dynamic_free.obs_buf)
            # shuffle(cbuf_dynamic_danger.obs_buf)            
            cbuf_dynamic_free.obs_buf = cbuf_dynamic_free.obs_buf[-N_DYNAMIC_BUFFER_FREE:]
            cbuf_dynamic_danger.obs_buf = cbuf_dynamic_danger.obs_buf[-N_DYNAMIC_BUFFER_DANGER:]            
            
            n_new_data = 0
            wandb.log({'time/training': time()-t0})
                
            # train_barrier(bnn, swa_bnn, boptimizer, 
            #               bbuf_traj, 
            #               cbuf_dynamic_free if train_dynamic_buffer else None, 
            #               cbuf_dynamic_danger if train_dynamic_buffer else None, 
            #               cbuf_agent, cbuf_obstacle, 
            #               n_iter=N_ITER) # int(total_trans//UPDATE_FREQ)

            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))
            
        if epoch_i < N_WARMUP:
            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_warmup.pt'))
                

        wandb.log({"loss/lr": boptimizer.param_groups[0]['lr'],
                   "uncollide_rates": np.mean(uncollide_rates),
                   "success_rates": np.mean(success_rates),
                   "running_unsafe_rate": running_unsafe_rate,
                   "explore_eps": explore_eps,
                   "nominal_eps": nominal_eps,
                   "relabel_prob": relabel_eps,
                   "no_feasible": no_feasible,
                   "volume": np.mean(volumes),
                   "n_trans": total_trans,
                   "n_relabels": n_relabels,
                   "n_evils": np.mean(n_evils),
                   "size/cbuf_obstacle": len(cbuf_obstacle),
                   "size/cbuf_agent": len(cbuf_agent),
                   "size/dybuf_free": len(cbuf_dynamic_free),
                   "size/dybuf_danger": len(cbuf_dynamic_danger),
                   "size/liebuf": len(bbuf_traj.dataset),
                   "size/gather": len(bbuf_gather.dataset),
                   "size/epoch_i": epoch_i})
        
