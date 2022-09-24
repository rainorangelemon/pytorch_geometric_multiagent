from tqdm import tqdm
import gc
from copy import deepcopy
from environment.gym_dubins_car import DubinsCarEnv, STEER

import faulthandler
faulthandler.enable()

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


from configs.v108 import *

Env = DubinsCarEnv
name_dict = generate_default_model_name(Env)
TXT_NAME = '1model_'+Env.__name__+'_'+version_name+'.txt'
BMODEL_PATH = name_dict['db'].replace('.pt', '_'+version_name+'.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def create_network():
    if ENV_CONFIG['PROB'][1] == 0:
        bnn = eval(MODEL)(HIDDEN_SIZE, keys=['agent'], pos_encode=PE_DIM)
    else:
        bnn = eval(MODEL)(HIDDEN_SIZE, keys=['agent', 'obstacle'], pos_encode=PE_DIM)
    bnn.to(device)
    return bnn


def create_env(num_agents=None, size=None, density=None, simple=None, min_dist=2):
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
        if size is not None:
            env_config['SIZE'] = (size, size)
        if density is not None:
            env_config['PROB'] = (0, density)
        if simple is not None:
            env_config['simple'] = simple
        while True:
            env = Env(**env_config)
            if (np.linalg.norm(env.world.agents[:,:env.space_dim] - env.world.agent_goals[:,:env.space_dim], axis=-1).min() >= min_dist):
                break
    return env


def train_barrier(bnn, swa_bnn, optimizer, buf, buf_traj, n_iter=10, relabel_eps=1):
    
    # Set up function for computing value loss
    def compute_bloss(bnn, swa_bnn, data, next_data):
        value = bnn(data)
        
        with torch.no_grad():
            tensor_a = torch.zeros(size=(len(data['agent'].x), n_candidates, data['action'].shape[-1]), device=device).uniform_(-1, 1)
            vec = swa_bnn.get_vec(next_data)
            vec = vec.unsqueeze(1).repeat((1, n_candidates, 1))
            next_bvalue = swa_bnn.get_field(vec, tensor_a)
        
        suspicious = (next_bvalue<DANGER_THRESHOLD).all(dim=-1).float()
        suspicious = (torch.cuda.FloatTensor(1).uniform_() < relabel_eps).float()*suspicious
        is_danger = ((suspicious + data['meet_obstacle'] + data['meet_agent'])>=1)
        bloss1 = (((1e-2-value).relu())*(~is_danger).float()).sum() / (1e-5 + (~is_danger).float().sum())
        bloss2 = (((1e-2+value).relu())*is_danger.float()).sum() / (1e-5 + is_danger.float().sum())
        bloss = bloss1 + bloss2
        wandb.log({"suspicious": suspicious.mean(),
                   "is_danger": is_danger.float().mean()})
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
            dloss = ((-deriv+1e-2).relu())*data['next_free']*next_data['next_free']
            dloss = dloss.mean()

        return dloss
    
    for i in range(n_iter):
        loader = buf.get()
        traj_loader = buf_traj.get()
        
        data, next_data = next(iter(traj_loader))
        optimizer.zero_grad()
        data = data.to(device)
        next_data = next_data.to(device)
        dloss = compute_dloss(bnn, data, next_data)
        dloss.backward()        
        torch.nn.utils.clip_grad_norm_(bnn.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"dloss": dloss})
        
        data, next_data = next(iter(loader))
        optimizer.zero_grad()
        data = data.to(device)
        next_data = next_data.to(device)
        bloss = compute_bloss(bnn, swa_bnn, data, next_data)
        bloss.backward()        
        torch.nn.utils.clip_grad_norm_(bnn.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"bloss": bloss,})
        
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

    def __init__(self):
        self.obs_buf = []       
        
    def store(self, obs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        expected keys: 'prev_free', 'next_free', 'prev_danger', 'next_danger', 'bvalue', 'action'
        """
        self.obs_buf.append(obs)
    
    
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
        self.batch = batch
        self.dataset = MyDataset([], [])
        
    def append(self, buffer, collided):
        if np.all(collided) and (not ALL_LIE):
            return
        
        new_buff = []
        for obs in buffer.obs_buf:
            clone_obs = obs.clone()
            clone_obs['next_free'] = (1-torch.FloatTensor(collided))*clone_obs['next_free']
            new_buff.append(clone_obs)
        
        for o, next_o in zip(new_buff[:-1], new_buff[1:]):
            self.dataset.data.append(o)
            self.dataset.next_data.append(next_o)
        
    def get(self):
        loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch, collate_fn=collate)
        return loader
    

@torch.no_grad()    
def eval_action(bnn, o, a, threshold, max_iter=30):
    # size of a: (num_agents, n_candidates, action_dim)
    n_candidate = a.shape[1]
    
    input_ = o.to(device)
    tensor_a = torch.FloatTensor(a).to(device)

    input_['action'] = tensor_a
    vec = bnn.get_vec(input_)
    vec = vec.unsqueeze(1).repeat((1, n_candidate, 1))
    bvalue = bnn.get_field(vec, tensor_a)
    return tensor_a.data.cpu().numpy(), bvalue.data.cpu().numpy()


def choose_action(bnn, env, o, explore_eps, nominal_eps, spatial_prop, threshold):
    if nominal_eps > 0:
        K1 = 1e-1
        K2 = -3e-2
    else:
        K1 = 0.
        K2 = -3e-2
    
    a_all = np.random.uniform(-1, 1, size=(env.num_agents, n_candidates, env.action_dim))
    a_refines, bvalues = eval_action(bnn, o, a_all, max_iter=0, threshold=threshold)

    dists = env.potential_field(a_refines, K1=K1, K2=K2, ignore_agent=(nominal_eps <= 0))
    v = np.zeros(env.num_agents)
    a = np.zeros((env.num_agents, env.action_dim))
    feasibles = np.zeros(env.num_agents)
    evil_agents = set()
    for agent_id, a_refine, bvalue, dist in zip(np.arange(env.num_agents), a_refines, bvalues, dists):

        if np.random.rand() < nominal_eps:
            a[agent_id] = a_refine[np.argsort(dist)[0]]
            feasibles[agent_id] = 1
            continue
        
        feasible_current = False
        if np.any(bvalue>threshold):
            feasible_current = True
        else:
            feasible_current = False
            
        if feasible_current:
            feasibles[agent_id] = 1
            for a_idx in np.argsort(dist):
                if bvalue[a_idx] > threshold:
                    a[agent_id] = a_refine[a_idx]
                    v[agent_id] = bvalue[a_idx]
                    break
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

                local_a_refines, local_bvalues = eval_action(bnn, local_o, a_all, max_iter=0, threshold=threshold)

                if np.any(local_bvalues[agent_id]>threshold):
                    evil_agents = evil_agents | local_evils
                    break
                    
        if np.random.rand()<explore_eps:
            a_idx = np.random.randint(n_candidates)
            a[agent_id] = a_refine[a_idx]
            if a_idx != np.argmax(bvalue):
                feasibles[agent_id] = 1  # mask the random action            
            continue
        else:
            a[agent_id] = a_refine[np.argmax(bvalue)]
            v[agent_id] = bvalue[np.argmax(bvalue)]

    for evil_agent in evil_agents:
        a_refine, bvalue = a_refines[evil_agent], bvalues[evil_agent]
        a[evil_agent] = a_refine[np.argmax(bvalue)]
        v[evil_agent] = bvalue[np.argmax(bvalue)]
        
    return a, v, feasibles, evil_agents


def save_gif(gifs, name="play.gif"):
    a_frames = []
    for img in gifs:
        a_frames.append(np.asarray(img))
    a_frames = np.stack(a_frames)
    ims = [Image.fromarray(a_frame) for a_frame in a_frames]
    ims[0].save(name, save_all=True, append_images=ims[1:], loop=0, duration=10)


@torch.no_grad()
def infer(env, bnn, threshold=None, max_episode_length=256, verbose=False, seed=0, stop_at_collision=False, spatial_prop=None, need_gif=True):
    if spatial_prop is None:
        spatial_prop = SPATIAL_PROP
    
    if threshold is None:
        threshold=THRESHOLD
    if verbose:
        print('----------------------------------------')
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if need_gif:
        gifs = [env._render()]
    else:
        gifs = [None]
    total_trans=0; n_danger=0; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool)

    while True:
        o = env._get_obs()
        a, v, feasibles, evil_agents = choose_action(bnn=bnn, env=env, o=o, explore_eps=0, 
                                                     nominal_eps=0, 
                                                     spatial_prop=spatial_prop, 
                                                     threshold=threshold)
        next_o, rw, done, info = env.step(a)
        
        prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
        next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
        if np.any(next_danger):
            collided = collided | next_danger
        if verbose:
            print(total_trans, np.where(v<=threshold), next_danger, evil_agents)
            
        total_trans += 1
        if need_gif:
            gifs.append(env._render())
        else:
            gifs.append(None)

        if np.any(next_danger) and stop_at_collision:
            break        
        
        if done or (total_trans >= max_episode_length):
            break

    return collided, done, gifs


if __name__ == '__main__':
    
    wandb.init(
        project="dubins_car",
        config={
            "env_config": ENV_CONFIG,
            "n_candidates": n_candidates,
            "batch": BATCH,
            "n_traj": N_TRAJ,
            "n_iter": N_ITER,
            "threshold": THRESHOLD,
            "n_traj_per_epoch": N_TRAJ_PER_EPOCH,
            "n_buffer": N_BUFFER,
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
            "only_boundary": ONLY_BOUNDARY,
            "polyak": POLYAK,
            "pe_dim": PE_DIM,
            "meet_danger": DANGER_THRESHOLD,
        },
        name=version_name,)
    
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
        env = create_env()
        valid_dataset.append((env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy()))


    Env = Env
    env = create_env()
    swa_bnn = bnn = create_network()

    name_dict = generate_default_model_name(Env)
    # bnn.load_state_dict(torch.load(name_dict['b'].replace('.pt', '_1model.pt'), map_location=device))

    if OPTIMIZER=='SGD':
        boptimizer = torch.optim.SGD(bnn.parameters(), lr=LR, momentum=0.9, weight_decay=1e-8)
    elif OPTIMIZER=='Adam':
        boptimizer = torch.optim.Adam(bnn.parameters(), lr=LR, weight_decay=1e-8)
    else:
        assert False
    if USE_SCHEDULER:
        bscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(boptimizer, factor=0.5, patience=PATIENCE)

    max_episode_length = 256

    running_unsafe_rate = 0
    best_unsafe_rate = float('inf')
    unsafe_rates = [1.]*N_EVALUATE
    uncollide_rates = [1.]*N_EVALUATE
    success_rates = [0.]*N_EVALUATE
    nominal_eps = 1.0
    explore_eps = 1.0

    trajs = defaultdict(list)
    t_all = 0
    bbuf_traj = TrajectoryReplayBuffer(BATCH)
    bbuf_gather = TrajectoryReplayBuffer(BATCH)

    for epoch_i in range(N_TRAJ):

        if epoch_i < N_WARMUP:
            explore_eps = 0.
        elif EXPLORE_WAY=='cyclic':
            explore_eps = (MAX_EXPLORE_EPS-(MAX_EXPLORE_EPS-MIN_EXPLORE_EPS)*((epoch_i-N_WARMUP) % 100)/100.)  # if ((epoch_i % 200) < 100) else 0.
        elif EXPLORE_WAY=='linear':
            explore_eps = np.clip(1. - 1.5 * ((epoch_i-N_WARMUP) // N_VALID) / (N_TRAJ // N_VALID), MIN_EXPLORE_EPS, MAX_EXPLORE_EPS)
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
            relabel_eps = (1-explore_eps)**2 # (epoch_i // N_VALID) / (N_TRAJ // N_VALID)
        else:
            relabel_eps = REFINE_EPS

        if (epoch_i % len(dataset) == 0):
            shuffle(dataset)

        bbuf = GlobalReplayBuffer()

        env = create_env()
        dataset[epoch_i%len(dataset)][-1] += 1; visit_time = 0
        if TRAIN_ON_HARD:
            if epoch_i > N_WARMUP:
                env.world.obstacles, env.world.agent_goals, env.world.agents, visit_time = deepcopy(dataset[epoch_i%len(dataset)])

        total_trans=0; n_danger=0; threshold=THRESHOLD; no_feasible=0; collided=np.zeros(env.num_agents).astype(bool); 
        volumes=[]; n_relabels=0; n_evils = []

        while True:
            o = env._get_obs()


            a, v, feasibles, evil_agents = choose_action(
                                            bnn=bnn,
                                            env=env, o=o, 
                                            explore_eps=explore_eps, 
                                            nominal_eps=max(nominal_eps, int(epoch_i<N_WARMUP)), 
                                            spatial_prop=SPATIAL_PROP,
                                            threshold=threshold,)
            n_evils.append(len(evil_agents))
            no_feasible += (env.num_agents - np.sum(feasibles))
            next_o, rw, done, info = env.step(a)

            info['feasible'] = torch.FloatTensor(feasibles)
            bbuf.store(info.clone())
            prev_danger = info['prev_danger'].data.cpu().numpy().astype(bool)
            next_danger = info['next_danger'].data.cpu().numpy().astype(bool)
            if np.any(next_danger):
                collided = collided | next_danger

            total_trans += 1
            t_all += 1
            n_danger += np.array(next_danger).sum()
            volumes.append((v>threshold).mean())

            if done or (total_trans >= max_episode_length):
                if (n_danger==0) or (visit_time > MAX_VISIT_TIME):
                    # only preserve the hard envs
                    dataset.pop(epoch_i%len(dataset))
                    env = create_env()
                    dataset.append([env.world.obstacles.copy(), env.world.agent_goals.copy(), env.world.agents.copy(), 0])

                bbuf_traj.append(bbuf, collided)
                bbuf_gather.append(bbuf, [False]*env.num_agents)
                break

        unsafe_rates.append(collided.mean())
        unsafe_rates.pop(0)
        uncollide_rates.append(np.any(collided))
        uncollide_rates.pop(0)    
        success_rates.append(done and (not np.any(collided)))
        success_rates.pop(0)
        running_unsafe_rate = np.mean(unsafe_rates)

        while len(bbuf_gather.dataset) > N_BUFFER:
            bbuf_gather.dataset.data.pop(0)
            bbuf_gather.dataset.next_data.pop(0)
            
        while len(bbuf_traj.dataset) > N_BUFFER:
            bbuf_traj.dataset.data.pop(0)
            bbuf_traj.dataset.next_data.pop(0)
            
        if (epoch_i % (N_VALID) == (N_VALID-1)):

            torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_{0:d}.pt'.format(epoch_i)))

            valid_loss = 0
            valid_success = 0
            valid_length = 0
            for v_idx, data in enumerate(valid_dataset):
                env = create_env()
                env.world.obstacles, env.world.agent_goals, env.world.agents = deepcopy(data)
                collided, done, gifs = infer(env, bnn, need_gif=False)
                valid_loss += np.mean(collided)
                valid_success += (done and (not np.any(collided)))
                valid_length += len(gifs)
            
            if USE_SCHEDULER and (explore_eps > MIN_EXPLORE_EPS):
                bscheduler.step(valid_loss/len(valid_dataset)+100*(1-valid_success/len(valid_dataset)))
            
            wandb.log({"valid loss": valid_loss/len(valid_dataset),
                       "valid length": valid_length/len(valid_dataset),
                       "valid success": valid_success/len(valid_dataset),})                       
        
        if (epoch_i % N_TRAJ_PER_EPOCH) == (N_TRAJ_PER_EPOCH-1) and (running_unsafe_rate!=0):
            if running_unsafe_rate < best_unsafe_rate:
                best_unsafe_rate = running_unsafe_rate
                torch.save(bnn.state_dict(), BMODEL_PATH)        

        if epoch_i == 9:
            swa_bnn = deepcopy(bnn)
            for p in swa_bnn.parameters():
                p.requires_grad = False            
            print(swa_bnn) 

        bnn.train()
        train_barrier(bnn, swa_bnn, boptimizer, bbuf_gather, bbuf_traj, n_iter=total_trans, relabel_eps=relabel_eps)

        torch.save(bnn.state_dict(), BMODEL_PATH.replace('.pt', '_current.pt'))
                

        wandb.log({"lr": boptimizer.param_groups[0]['lr'],
                   "uncollide_rates": np.mean(uncollide_rates),
                   "success_rates": np.mean(success_rates),
                   "running_unsafe_rate": running_unsafe_rate,
                   "explore_eps": explore_eps,
                   "nominal_eps": nominal_eps,
                   "relabel_prob": relabel_eps,
                   "no_feasible": no_feasible,
                   "dataset_size": len(bbuf_traj.dataset),
                   "volume": np.mean(volumes),
                   "n_trans": total_trans,
                   "n_relabels": n_relabels,
                   "n_evils": np.mean(n_evils),})
        
